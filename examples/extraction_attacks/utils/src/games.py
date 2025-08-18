import numpy as np
from  .players import TsallisInf
import abc
import examples.extraction_attacks.utils.pool_utils as pool_utils
from torch import no_grad, softmax, tensor
from leakpro.attacks.utils.utils import softmax_logits
from examples.mia.text_mia.utils.tabds_data_preparation import LabelSet

class Game:
    """A class used to represent a leader-follower game"""
    def __init__(self, T, num_players, num_player_actions):
        self.T = T # number of rounds
        self.num_players = num_players # number of players
        self.num_player_actions = num_player_actions # number of actions
        self.players = [] # list to store player objects
        self.create_players() # populate the list with players taking into account the leader-follower hierarchy
    
    def create_players(self):
        """Create players based on leader-follower structure"""
        for i in range(self.num_players):
            self.players.append(TsallisInf(self.num_player_actions[i])) # create a player that is aware of the actions players going before may take
    
    def get_player_actions(self):
        """Sample an action from all the players"""
        actions = [] # list to store the actions
        for i in range(self.num_players):  # go through all the players
            actions.append(self.players[i].sample_action())
        return actions # return the joint action
    
    def reset_game(self):
        """Reset the game"""
        for player in self.players: # reset each player
            player.reset()
    
    @abc.abstractmethod
    def step(self):
        """Abstract method to play the game for one round"""
        return
            
    @abc.abstractmethod
    def get_reward(self, action):
        """Abstract method that is game specific"""
        return

class SimpleGame(Game):
    def __init__(self, T):
        num_players = 2
        num_player_actions = [10,10]
        self.rewards = {0: np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                        1: np.array([0.3, 0.3, 0.2, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0])}
        super().__init__(T, num_players, num_player_actions)
    
    def get_reward(self, action):
        # inpaint document with action indices
        # run inpainted document through the model
        # obtain rewards
        
        reward = [-1, -1]
        for i in range(self.num_players):
            reward[i] = self.rewards[i][action[i]]
        return reward
        
    def step(self):   
        """Step through one round of the game"""
        # sample actions
        actions = super().get_player_actions()
        observed_reward = self.get_reward(actions)
        return observed_reward
    
    def update_policies(self, rewards, time):
        """Update players policies based on observed reward"""
        for i in range(self.num_players):  
            self.players[i].update_policy(rewards[i], time)


class TsallisGame(Game):

    
    def __init__(self, T, n_masks, mask_types, pool, attack_obj):
        num_players = n_masks # One player per mask
        self.attack_attempt = 0 # Keep track of how many guesses whe've made
        #self.pool = pool.get_pool() # Get the pool of data Not used here so probably can remove
        self.pool_sizes = pool.get_pool_sizes() # A dict with len of each entity type
        self.global_pool_bool = pool.global_attack_pool_bool
        self.label_set = LabelSet(attack_obj.handler.configs["audit"]["class_list"]) # A set with the labels
        self.mask_types = mask_types # A list with the entity types of each player
        
        num_player_actions = [] # The actions taken by the player
        self.reward_factor = attack_obj.tsallis_reward_factor # The reward factor, unclear benefit
        self.attack_obj = attack_obj # MMattack object
        self.correct_idxs = pool.true_idxs # True indexes of the masks in the pools
        
        class_masking = True # If we mask by class type, or only [MASK]
        if class_masking:
            raw_labels=['PERSON', 'CODE', 'LOC', 'ORG', 'DEM', 'DATETIME', 'QUANTITY', 'MISC']
        else:
            raw_labels = ['MASK']

        # Create dictionaries to convert from the labels to integers, and back. -1 for unmasked
        self.conversion_dict = {i: label for i, label in enumerate(raw_labels)}
        self.conversion_dict[-1] = -1
        self.reverse_conversion_dict = {v: k for k, v in self.conversion_dict.items()}
        
        # Initiate the pool sizes for the cases where the attack pool is global or local
        if not self.global_pool_bool:
            for i in range(num_players):
                num_player_actions.append(self.pool_sizes[i])
        else:     
            for mask in mask_types:
                num_player_actions.append(self.pool_sizes[mask.item()]) # number of actions for each player
       
        self.n_matches = np.zeros(T) # To use later, keep track of how many correct guesses per attempt
        super().__init__(T, num_players, num_player_actions) # Initiate the game
    
    def get_reward(self, action):
        # inpaint document with action indices
        # run inpainted document through the model
        # obtain rewards

        dev = self.attack_obj.device
        
        selected_pii_actions = []
        for i in range(self.num_players):
            mask_id = self.mask_types[i]
            selected_pii_actions.append(action[i]) #NOTE Check that this is correctly done.

        self.attack_obj.actions_taken[self.attack_attempt] = selected_pii_actions
        
        n_correct_guesses = sum( int(x == y) for x, y in zip(selected_pii_actions, self.correct_idxs) )

        self.n_matches[self.attack_attempt] = n_correct_guesses

        # Inpaint the document with the selected actions 
        # filled_token_ids is a list where each element is the tokens filled for each respective position
        filled_doc, filled_token_ids = self.attack_obj.fill_blanks(indices_of_masks = self.attack_obj.adjusted_indices_of_target_masks,
                                                                          action_list = selected_pii_actions)
        
        # Keep track of the selected tokens for each guess (TODO Maybe could move this to be related to the pools instead?)
        self.attack_obj.guessed_tokens[self.attack_attempt] = filled_token_ids
        
        target_model = self.attack_obj.target_model.to(dev)
        target_model.eval()
        
        
        # Evaluate the inpatinted document, and save the rewards
        with no_grad():
            
            
            #Extract the input ids and masks
            x_input_ids = filled_doc['input_ids'].unsqueeze(0).to(dev)
            x_attention_mask = filled_doc['attention_masks'].unsqueeze(0).to(dev)
            
            # Run through the model
            logits_target = target_model(input_ids = x_input_ids, attention_mask = x_attention_mask)

            # Calculate the softmax logits of the received logits
            probs_target = softmax_logits(logits_target.cpu().numpy())
            probs_of_masks = self.attack_obj.calculate_confidence_for_masked_tokens(probs = probs_target, 
                                                                                    i_guess=self.attack_attempt)
            self.attack_obj.confidence_scores[self.attack_attempt] = probs_of_masks
   

        reward = [probs_of_masks[i]**self.reward_factor for i in range(self.num_players)]
 
        
        return reward
        
    def step(self):   
        """Step through one round of the game"""
        
        # Choose actions
        actions = super().get_player_actions()
        # Get rewards
        observed_reward = self.get_reward(actions)
        
        self.attack_attempt += 1
        return observed_reward
    
    def calculate_correct_choices(self, actions):
        """Counts how many of the chosen actions were the correct ones"""
        # TODO fix this function to get simpler access to number of correct choices
        
        self.n_matches[self.attack_attempt] = sum(1 for x, y in zip(actions, self.correct_idxs) if x == y)
        
    
    def update_policies(self, rewards, time):
        """Update players policies based on observed reward"""
        for i in range(self.num_players):  
            self.players[i].update_policy(rewards[i], time)
