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


class ExtractionGame(Game):
    def __init__(self, T, n_masks, mask_types, pool, attack_obj):
        num_players = n_masks # One player per mask
        self.attack_attempt = 0
        self.pool = pool.get_pool() # Get the pool of data
        self.pool_sizes = pool.get_pool_sizes() # A dict with len of each entity type
        self.label_set = LabelSet(attack_obj.handler.configs["audit"]["class_list"]) # A set with the labels
        self.mask_types = mask_types # A list with the entity types of each player
        num_player_actions = []

        for i in range(n_masks):
            mask_id = self.label_set.ids_to_label[mask_types[i].item()][2:]
            num_player_actions.append(self.pool_sizes[mask_id]) # number of actions for each player
        
        self.n_matches = np.zeros(T)
        self.attack_obj = attack_obj
        super().__init__(T, num_players, num_player_actions)
    
    def get_reward(self, action):
        # inpaint document with action indices
        # run inpainted document through the model
        # obtain rewards
        
        selected_pii_actions = []
        for i in range(self.num_players):
            mask_id = self.mask_types[i]
            selected_pii_actions.append(action[i])

        # run the model with the selected pii imputed 
        # and get the reward
        attack_doc_member, token_ids_member = self.attack_obj.fill_blanks(self.attack_obj.trimmed_member_example, 
                                                                          self.attack_obj.n_tokens_to_fill_member)
        self.attack_obj.tokens_member_doc[self.attack_attempt] = token_ids_member
        
        target_model = self.attack_obj.target_model.to(self.attack_obj.device)
        target_model.eval()
        
        with no_grad():
            
            member_feats, member_labels  = self.attack_obj.custom_collate_fn(attack_doc_member)

            logits_member = target_model(member_feats)
            probs_member = softmax_logits(logits_member.cpu().numpy())
            probs_of_masks = self.attack_obj.calculate_confidence_for_masked_tokens(probs = probs_member, 
                                                                                    member_doc = True, 
                                                                                    i_guess=self.attack_attempt)
            self.attack_obj.confidence_scores_member[self.attack_attempt] = probs_of_masks


        reward = [0 for i in range(self.num_players)]
        reward_factor = 1
        for i in range(self.num_players):
            reward[i] = probs_of_masks[i]**reward_factor

        return reward
        
    def step(self):   
        """Step through one round of the game"""
        # sample actions
        actions = super().get_player_actions()
        #self.calculate_correct_choices(actions)
        observed_reward = self.get_reward(actions)
        self.attack_attempt += 1
        return observed_reward
    
    def calculate_correct_choices(self, actions):
        """Counts how many of the chosen actions were the correct ones"""
        self.n_matches[self.attack_attempt] = sum(1 for x, y in zip(actions, correct_ids) if x == y)
        
    
    def update_policies(self, rewards, time):
        """Update players policies based on observed reward"""
        for i in range(self.num_players):  
            self.players[i].update_policy(rewards[i], time)
