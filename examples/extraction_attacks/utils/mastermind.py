"""Implementation of the Mastermind extraction attack."""
import numpy as np
from datetime import datetime
from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from examples.extraction_attacks.utils.pool_utils import *
from examples.mia.text_mia.utils.tabds_data_preparation import TrainingExample, Batch
from transformers import LongformerTokenizerFast, PreTrainedTokenizerFast
from torch import device, optim, cuda, no_grad, save, sigmoid, Tensor, zeros, tensor, ones, long, argmax, cat
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from examples.extraction_attacks.utils.src.games import TsallisGame
from tqdm import tqdm  
import matplotlib.pyplot as plt 
from utils.plot_utils import *

MODEL_MAX_LEN = 4096

class AttackMM(AbstractMIA):
    
    def __init__(self: Self, handler: AbstractInputHandler, configs: dict, pool: PIIPool = None) -> None:
        #DONE
        super().__init__(handler)
        self.pool = pool
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.tokenizer = self.handler.population.tokenizer
        self.attack_attempt = 0
        self._configure_attack(configs)



    def _configure_attack(self, configs) -> None:
        """ 
        Get all the configs from the audit file needed to run the attack. 
        """
        
        # General configs
        self.attack_strategy = configs.get("attack_strategy", "random") # "random" is default
        self.n_evaluations = configs.get("n_filling_attempts", 10) # How many times we try to impute the document with text according to the strategy
        self.target_doc_idx = configs.get("target_doc_idx", 0) # Set the target document to attack TODO: Change this to be more flexible maybe
        self.exact_matching_confidence = configs.get("exact_matching_confidence", False) # Either matches only B and I labels, or both 
        
        
        # Pool specifics 
        self.data_pool_path = configs.get("data_pool_path", None) # For the pool of the data
        self.global_attack_pool_bool = configs.get("global_attack_pool", True) # If the pool is global for all positions or local
        self.attack_pool_type = configs.get("attack_pool_type", "full") # full or extended
        self.attack_pool_size_extension = configs.get("attack_pool_size_extension", 0) # How many extra entities to add to the attack pool, aside from the correct ones
        
        # Bandit specifics
        self.bandit_parameters = configs.get("bandit_parameters", None) # Parameters for the bandit strategy
        self.bandit_sampling_alg = configs.get("bandit_alg", "tsallis") # The bandit sampling algorithm to use
        
        if self.bandit_parameters and self.bandit_sampling_alg == "tsallis":
            self.tsallis_reward_factor = self.bandit_parameters['tsallis_reward_factor']

        self.multi_run_averaging = configs.get('multi_run_averaging', False)

        if self.multi_run_averaging:
            self.attack_strategy = "random"
            multi_run_parameters = configs.get('multi_run_parameters', None)
            self.MR_n_docs = multi_run_parameters['n_docs']
            self.MR_randomize_doc_ids = multi_run_parameters['randomize_doc_ids']
            self.MR_entity = multi_run_parameters['entity_type_id']
            # Maybe like this, maybe have it evenly spaced out between 0 and 1 in n_runs_per_doc steps
            # Currently evenly spaced.
            self.MR_set_fraction = multi_run_parameters['correct_fraction'] 
            self.MR_correct_fractions = []
            self.MR_actual_correct_fractions = []

            self.n_runs_per_doc = self.n_evaluations
            self.MR_docs_used = []
            self.MR_counter = -1


        self.population_length = len(self.handler.population)

       

        # Limit the nr of masks in the document (i.e. leave the first n_mask limit only, with the rest unmasked)
        self.limit_masks = configs.get("limit_masks", False) # If there is a limit on the number of masks censor and fill
        self.n_mask_limit = configs.get("n_mask_limit", 5) # The number of masks to limit to in that case


        # Save some specifics of a single mask, which is always correct
        self.single_investigation = configs.get("single_investigation", False)
        self.SI_mask_idx = configs.get("single_mask_idx", 0)
        self.SI_replace = False
        self.SI_replacement_done = False # If we have already replaced the mask with a random one

        # Expanding window parameters
        self.expanding_window = configs.get('expanding_window', False) # Run this experiment
        self.EW_entity_type_int = configs.get('ew_entity_type_id', None) # The entity type used for the experiment.
        
        # Parameters related to investigating only a single mask rather than a specific entity type
        self.EW_single_entity_bool = configs.get('ew_single_entity_bool', False) # If true, we'll check over a specific entity. If false, we check over all entities of the specified class.
        self.EW_single_entity_random_mask_bool = configs.get('EW_single_entity_random_mask_bool', False) # If true, we will sample a random target mask for the single entity case
        self.EW_single_entity_mask_idx = configs.get('EW_single_entity_mask_idx', 0) # This is the per entity mask index, K will correspond to the Kth mask of the entity type investigated
        self.EW_single_entity_token_idx_in_org = -1         # The token index in the true document (initialized )
        self.EW_single_entity_token_idx_in_trimmed = -1     # The token index in the sequence of the trimmed document (initialized here)
        self.EW_single_entity_actual_mask_idx = -1          # The masks order over all entity types
        self.EW_single_entity_org_mask_len = -1             # Length of the specified mask
        self.EW_single_entity_replace_target_mask_bool = configs.get('ew_single_entity_replace_target_bool', False) # Impute with a random mask
        self.EW_single_entity_replaced_mask_len = -1        # Length of the false mask imputed
        self.EW_single_entity_set_rest_correct_bool = configs.get('EW_single_entity_set_rest_correct_bool', True) # What to do with the rest of the masks. 



        self.EW_centered_single = configs.get('ew_centered_single', False) # The window expands from the center of the single entity.
        if self.EW_centered_single: 
            self.EW_single_entity_bool = True # Overrides the single entity setting, as the centered single entity is always a single entity investigation
            self.EW_centered_offset = zeros(self.n_evaluations, dtype = int)    # List of used window sizes
            self.EW_centered_offset[0] = 10                                     # Initialize the window size
            self.EW_centered_offset_increase = 10                               # The increase of the window size
            self.EW_centered_single_current_token_idx = -1                      # The masks token index in the current slice of the document



        # Gather context free confidences of the masks in all of the training data
        self.gather_context_free_statistics = configs.get('gather_context_free_statistics', False) # Runs the model on context-free masks from the training data
            
        self.path_to_output = './outputs/'
        self.experiment_path = './outputs/'  + datetime.today().strftime('%y%m%d_%H%M') + '_' + self.attack_strategy
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

        self.figpath = self.experiment_path +'/figures'  
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)

        if self.multi_run_averaging:
            multi_run_path = self.experiment_path + '/multi_run_results'
            if not os.path.exists(multi_run_path):
                os.makedirs(multi_run_path)
   
        

    def prepare_attack(self, target_doc_idx = None) -> None:
        
        # Enables reusing this code for multi-run averaging
        if self.multi_run_averaging:
            if not self.MR_randomize_doc_ids:
                self.target_doc_idx = target_doc_idx

        # Retrieve the features and labels of the target document         
        self.target_example = self.handler.population[self.target_doc_idx]
      
        # Set the tokenizer and target model
        self.target_model = self.handler.target_model
        self.doc_token_length = sum(self.target_example['attention_masks']).item()
  
        # Dictionaries to convert identifier and entity type between int and string
        # (TODO Move this to the dataset itself maybe)
        self.identifier_string_to_int = {
            'O': 0,
            'DIRECT': 1,
            'QUASI': 2
        }
        self.identifier_int_to_string = {v: k for k, v in self.identifier_string_to_int.items()}
        
        
        self.entity_string_to_int  = {
            'O': -1,
            'PERSON': 0,
            'CODE': 1,
            'LOC': 2,
            'ORG': 3,
            'DEM': 4,
            'DATETIME': 5,
            'QUANTITY': 6,
            'MISC': 7
        }

        self.entity_int_to_string = {v: k for k, v in self.entity_string_to_int.items()}
    
        # Set the target features and labels, and convert the strings to integers for e_t and i_t
        self.target_features = {
            'input_ids': self.target_example['input_ids'], 
            'attention_masks': self.target_example['attention_masks'],
            'entity_types': tensor(self.target_example['entity_types'], dtype = int),
            'identifier_types': tensor([self.identifier_string_to_int[i] for i in self.target_example['identifier_types']], dtype = int)
        }
        self.target_labels = self.target_example['labels'] # Why is this not in the features dict? FIXME mayb

        # Generate the trimmed target example and some info about it
        self.trim_example() 
        
        # Adjust the indices of masks in the trimmed example, as we're removing tokens the latter indices need updating
        adjusted_masks = np.append([0], np.cumsum(self.len_of_target_masks)[:-1])
        adjusted_masks-= np.arange(self.n_masks_in_target)
        self.adjusted_indices_of_target_masks = self.indices_of_target_masks - adjusted_masks
        
        # Extract the true input ids and entity types for the target document, used in evaluation
        # Input ids are the tokens, entity types here are ints
        self.true_input_ids = [self.target_features['input_ids'][idx:idx + self.len_of_target_masks[i]] for i, idx in enumerate(self.indices_of_target_masks)]
        self.true_entity_types = np.array([self.trimmed_target_example['entity_types'][i] for i in self.adjusted_indices_of_target_masks])
      
        if self.expanding_window:
            
            indices_of_entity_type = np.where(self.true_entity_types == self.EW_entity_type_int)[0]
            
            if self.EW_single_entity_bool:
                
                assert len(indices_of_entity_type) > self.EW_single_entity_mask_idx, "Didn't find the mask with desired position in the PII Class"
                
                single_entity_mask_idx = indices_of_entity_type[self.EW_single_entity_mask_idx]
                
                if self.EW_single_entity_random_mask_bool: # Randomizes which masked position is selected
                    
                    remaining_indices = indices_of_entity_type[ indices_of_entity_type != single_entity_mask_idx]
                    single_entity_mask_idx = np.random.choice(remaining_indices)

             
                self.EW_single_entity_actual_mask_idx = single_entity_mask_idx # The mask idx of the single entity, i.e. which idx of the masks is the sought after one
                self.EW_single_entity_token_idx_in_trimmed = self.adjusted_indices_of_target_masks[single_entity_mask_idx] # Token idx in the trimmed document
                self.EW_single_entity_token_idx_in_org = self.indices_of_target_masks[single_entity_mask_idx] # Token idx in the original document
                self.EW_single_entity_org_mask_len = self.len_of_target_masks[single_entity_mask_idx] # How long is the single PII we're investigating


        # Initialize some arrays used in the attack
        # Confidence scores (rewards)
        self.confidence_scores = [[] for i in range(self.n_evaluations)]

        # Guessed token
        self.guessed_tokens = [[] for i in range(self.n_evaluations)]
        
        # Actions taken
        self.actions_taken = [[] for i in range(self.n_evaluations)]

        # One hot encoding of correct actions per mask
        self.binary_correct_guesses = np.zeros(self.n_masks_in_target, dtype = int)
        self.n_corr_per_guess = np.zeros(self.n_evaluations, dtype=int)

        # Number of correct guesses
        self.final_n_correct = 0
        self.final_n_almost_correct = 0
        self.attack_attempt = 0


        # Create the attack pool if it doesn't exist
        if self.pool is None or self.multi_run_averaging:
            if self.data_pool_path is None:
                raise ValueError("No datapool path provided")
            self.pool = PIIPool(self.data_pool_path, 
                                self.tokenizer, 
                                self.global_attack_pool_bool, 
                                self.attack_pool_type, 
                                self.attack_pool_size_extension)


        self.pool.create_attack_pool(list(zip(self.true_input_ids, self.true_entity_types)))
        self.true_conf = self.get_true_confidences()
        
        # Specific related to single mask investigation
        if self.single_investigation:
            self.single_update_freq = self.n_masks_in_target / self.n_evaluations
            self.single_res = np.zeros((2, self.n_evaluations)) # (confidence score, fraction of correct masks) per evaluation


        if self.multi_run_averaging:
            self.MR_correct_fractions = np.arange(0, 1, 1/self.n_evaluations) # Just a linear fraction between 0 and 1, will be altered later on to reflect the true fractions implemented
            self.attack_attempt = 0
            self.MR_actual_correct_fractions = []
            self.MR_counter += 1
        
        
        if self.expanding_window:
            
            self.EW_original_size = 100     # Set the window original window size
            self.EW_current_size = 100      # Initialize the tracker of the window size
            self.EW_current_end = 0         # The final included index in the original document  
            self.EW_n_increments = 10       # How many increments do we wish to have? 
            self.EW_increment_size = ( self.doc_token_length - self.EW_original_size)/self.EW_n_increments # How big are the steps of increments? 
            self.EW_freq_increments = int( self.n_evaluations / self.EW_n_increments )
            self.EW_size_history = np.zeros(self.n_evaluations)
        
            if self.EW_single_entity_replace_target_mask_bool: 
                    
                    # Sample another entity which is replacing the true mask, different from replace mask idx since that replaces position, this replaces the content of the mask
                    self.EW_replacement_sequence = self.pool.sample_from_candidates(self.EW_entity_type_int, entity_id = -1, local_pool_id = self.EW_single_entity_actual_mask_idx)
                    self.EW_replacement_sequence_length = len(self.EW_replacement_sequence)

            if self.EW_centered_single: 
    
                max_edge_distance_from_token = np.max([ self.EW_single_entity_token_idx_in_org, self.doc_token_length -  self.EW_single_entity_token_idx_in_org ])
                self.EW_centered_offset_increase = int(max_edge_distance_from_token / self.n_evaluations)
                self.EW_centered_single_current_token_idx = self.EW_centered_offset[0] # The current token idx in the centered single mask attack
         
             



    def run_attack(self):
        """
        Main attack runner. Depending on the settings executes the different types of extraction attacks. 
        Or gathers statistics
        """
        if self.gather_context_free_statistics:
            logger.info('Gathering statistics from the training data')
            self.gather_statistics()
            logger.info('Gathering statistics done.')
            return


        # Random attack
        if self.attack_strategy == "random":

            logger.info('Using the random sampling strategy.')
            
            if self.multi_run_averaging:
                
                logger.info('Multi run averaging begins')

                for i in range(self.MR_n_docs):
                
                    self.run_multi_attack() 
                    logger.info(f'Multi run iteration {i+1}/{self.MR_n_docs} completed.')
                
                logger.info('Multi run averaging done!')
            
            else: 
            
                self.run_random_attack()

                logger.info("Random Attack Iterations Completed")

        # Bandit Attack TODO: Might need changing if more bandit algorithms are added
        elif self.attack_strategy == "bandit": 

            logger.info("Using bandit approach")
            
            if self.bandit_sampling_alg == "tsallis":
            
                logger.info("Using Tsallis inf sampling strategy.")
                self.run_tsallis_attack()
        
            logger.info("Bandit Iterations Completed")
        
        
        
        self.generate_results()
        logger.info("Results generated")


    def run_random_attack(self):
        """
        Random attack head. Fills in the tokens with random masks from the pools and evaluates. 
        If Expanding window is used, there's a difference in that the filled masks are the correct ones, for either specific or all masks. 
        """

        lock_top = False

        self.target_model.to(self.device)
        self.target_model.eval()

        for iAttempt in range(self.n_evaluations):
            
            
            if self.expanding_window:
                filled_target = self.generate_EW_filled_target(iAttempt)
            
            elif self.single_investigation:
                filled_target = self.generate_single_mask_filled_target(iAttempt)
            
            else: 
            
                filled_target, guessed_token_ids = self.fill_blanks( indices_of_masks = self.adjusted_indices_of_target_masks, 
                                                        action_list = None )
                
                self.guessed_tokens[iAttempt] = guessed_token_ids

            self.calculate_confidence_for_filled_target(iAttempt, filled_target)
            
            if lock_top: 
                #Not used 
                if iAttempt % 10 == 0:
                    self.lock_guess(iAttempt)



    def run_multi_attack(self):
        self.multi_run_target_doc_idx = np.random.randint(self.population_length)
        self.MR_docs_used.append(self.multi_run_target_doc_idx)
        self.run_random_attack()
        self.save_multi_run_results()
        self.prepare_attack()  


    def generate_single_mask_filled_target(self, iAttempt):


        """
        Generates the filled target for the single mask investigation. 
        Fills in the mask with the correct token, and randomizes the rest of the tokens.
        """
        
       
        single_mask_idx = self.indices_of_target_masks[self.SI_mask_idx] # The token idx in the original document
        single_mask_len = self.len_of_target_masks[self.SI_mask_idx] # The length of the mask in the original document
        single_mask_label = self.target_labels[single_mask_idx] # The label of the mask in the original document
        # Fill in the mask with the correct token

        pre_mask = self.target_features['input_ids'][:single_mask_idx]
        post_mask = self.target_features['input_ids'][single_mask_idx + single_mask_len:]
        
        if self.SI_replace:
            mask_seq = self.pool.sample_from_candidates(self.EW_entity_type_int, entity_id=-1, local_pool_id=self.SI_mask_idx)
            self.SI_replaced_mask_length = len(mask_seq)
        else:
            
            # Mask sequence is the correct token for the mask
            mask_seq = self.true_input_ids[self.SI_mask_idx] # The correct token for the mask

        filled_target = {
            'input_ids': cat([tensor([0], dtype=int), pre_mask, mask_seq, post_mask, tensor([2], dtype=int)]), # Add start and end tokens
            'attention_masks': cat( [ones(len(pre_mask) + len(mask_seq) + len(post_mask) + 2, dtype=int), zeros(MODEL_MAX_LEN  - (len(pre_mask) + len(mask_seq) + len(post_mask) + 2), dtype=int)]) # Add attention masks
        }
        
      
        return filled_target
    


    def calculate_confidence_for_filled_target(self, iAttempt, filled_target):
        """
        Given a filled document, this function calls for the calculation of the confidences.
        """
        
        with no_grad():
          
            logits = self.target_model( filled_target['input_ids'].unsqueeze(0).to(self.device),
                                            filled_target['attention_masks'].unsqueeze(0).to(self.device) )
            
            target_model_confidence = softmax_logits( logits.cpu().numpy() )
            confidence_of_masks = self.calculate_confidence_for_masked_tokens( confidences = target_model_confidence, i_guess = iAttempt )
            self.confidence_scores[iAttempt] = confidence_of_masks
            self.observed_rewards = self.confidence_scores # Differs in case of bandit strategies are used, but the same here


    def lock_guess(self, p_lock: int):
        """ 
        Function to lock the the token at ordered position up to, and including, p_lock. 
        NOTE Not used, not done.
        """

        assert p_lock <= self.n_masks_in_target, 'Attempt to lock more masks than available'

        best_action_at_p = np.argmax(self.observed_rewards[:][p_lock])
        self.trimmed_target_example['input_ids'][self.adjusted_indices_of_target_masks[p_lock]] = self.guessed_tokens[best_action_at_p][p_lock]
    




    def get_true_confidences(self):
        """
        Sends the full target document to the model and calculates and returns the true confidences.
        """

        target_model = self.target_model.to(self.device)
        target_model.eval()
        
        with no_grad():
        
            true_logits = target_model(self.target_example['input_ids'].unsqueeze(0).to(self.device),
                                        self.target_example['attention_masks'].unsqueeze(0).to(self.device))
            true_confidences = softmax_logits(true_logits.cpu().numpy())
        
        confidences_of_masks = self.calculate_confidence_for_masked_tokens( confidences = true_confidences, i_guess = -1 )

        return confidences_of_masks
        
        

    def run_tsallis_attack(self):
        """
        Runs the Tsallis-inf bandit algorithm. 
        Creates the game and plays the rounds.
        """
        self.extraction_game = TsallisGame(T = self.n_evaluations, 
                                            n_masks = self.n_masks_in_target, 
                                            mask_types = self.true_entity_types, 
                                            pool = self.pool,
                                            attack_obj = self )
        
        self.observed_rewards = []
        self.extraction_game.reset_game()
        
        
        for t in tqdm(range(self.n_evaluations)):
        
            self.observed_rewards.append(self.extraction_game.step())
            self.extraction_game.update_policies(self.observed_rewards[t], t)



    def generate_EW_filled_target(self, iAttempt):
        """
        Creates the filled target in the Expanding Window approach. 
        For each attack attempt, calculates the size of the window surrounding the mask, returns the relevant piece of the target document. 
        If the other tokens are to be randomized, they are also filled. 
        NOTE doesn't work as intended with the randomization currently i just realized, but since we're not investigating that right now it's fine. 
        
        """
        
        

        if self.EW_centered_single:
            
            window_start, window_end, window_len = self.adjust_centered_window_size(iAttempt)
        

        else: 

            window_start, window_end, window_len = self.adjust_right_increasing_window_size(iAttempt)

        if self.EW_single_entity_set_rest_correct_bool: 

            pass # Here we should implement some stuff which randomizes the rest of the tokens in the sequence. 
        
        pre_mask = self.target_features['input_ids'][ window_start : self.EW_single_entity_token_idx_in_org ]
        post_mask = self.target_features['input_ids'][ self.EW_single_entity_token_idx_in_org + self.EW_single_entity_org_mask_len : window_end + 1 ]
        
        if self.EW_single_entity_replace_target_mask_bool: # Replace single mask with another one
            
            mask_seq = self.EW_replacement_sequence

        
        else: 
            
            mask_seq = self.true_input_ids[self.EW_single_entity_actual_mask_idx]
        
        padding_to_add = MODEL_MAX_LEN  - (len(pre_mask) + len(mask_seq) + len(post_mask) + 2) # 2 for the start and end tokens
        
        filled_target = {
                            'input_ids': cat([tensor([0], dtype = int), pre_mask, mask_seq, post_mask , tensor([2], dtype = int), ones(padding_to_add, dtype = int)]), # Add start and end tokens
                            'attention_masks': cat([ones(MODEL_MAX_LEN  - padding_to_add, dtype = int), zeros(padding_to_add, dtype = int)]) 
                        }     


        assert filled_target['input_ids'].shape[0] == MODEL_MAX_LEN  and filled_target['attention_masks'].shape[0] == MODEL_MAX_LEN , f'Dimension mismatch ids {filled_target["input_ids"].shape[0]}, ams {filled_target["attention_masks"].shape[0]}'
        
        self.attack_attempt += 1
            
        return filled_target
    

    def adjust_right_increasing_window_size(self, iAttempt):
        
        
        if iAttempt % self.EW_freq_increments == 0 and iAttempt > 0: # Increase the expanding window size peroidically 
            self.EW_current_size = self.EW_original_size + self.EW_increment_size * ( iAttempt // self.EW_freq_increments)
            self.EW_current_size = int( min(MODEL_MAX_LEN , self.EW_current_size) ) 
        
        self.EW_size_history[iAttempt] = self.EW_current_size
        self.EW_current_end = self.EW_current_size

        window_start = 1
        window_end = self.EW_current_end
        window_len = window_end - window_start


        return window_start, window_end, window_len


    def adjust_centered_window_size(self, iAttempt):

        if iAttempt > 0: 
            self.EW_centered_offset[iAttempt] = self.EW_centered_offset[iAttempt-1] + self.EW_centered_offset_increase 
        
     
        
        window_start = int(max(1, self.EW_single_entity_token_idx_in_org - self.EW_centered_offset[iAttempt]))
        
        if self.EW_single_entity_replace_target_mask_bool:

            window_end = int(min(4095, self.EW_single_entity_token_idx_in_org + self.EW_single_entity_replaced_mask_len + self.EW_centered_offset[iAttempt]))
        
        else:
            # If not replacing the mask, use the original mask length
            window_end = int(min(4095, self.EW_single_entity_token_idx_in_org + self.EW_single_entity_org_mask_len + self.EW_centered_offset[iAttempt]))

        self.EW_current_end = window_end
        window_len = window_end - window_start
        self.EW_size_history[iAttempt] = window_len
        
        if window_start == 1: # Set the current token idx in the window used
        
            self.EW_centered_single_current_token_idx = self.EW_single_entity_token_idx_in_org
        
        else: 
        
            self.EW_centered_single_current_token_idx = self.EW_centered_offset[iAttempt] + 1
        
        return window_start, window_end, window_len


    def calculate_confidence_centered(self, confidences):
        """
        Calculate the confidence scores for the masked tokens for the case where there's an expanding window centered around
        a specific mask. Reason for existing is that the mask index will be different in between runs.
        Here the array 'probs' is only the probabilities related to the single mask which is being investigated.

        """

        
        mask_len = len(confidences)
        text_doc = self.target_example
        res = np.zeros(1)

        # Get the correct classes of the masked tokens
       

        org_idx = self.EW_single_entity_token_idx_in_org
        token_label = text_doc['labels'][org_idx]
       
        if token_label % 2 == 0: 
            other_label = -1 
        else: 
            other_label = 1 
        
        curr_idx = self.EW_centered_single_current_token_idx
        curr_idx = 0

        # Sum the token probabilities of the entire length of the masked sequence, for both B- and I- labels
        for j in range(mask_len):
 
            # Add the probabilities for this masked sequence, for both B- and I- labels
            if not self.exact_matching_confidence:

                res += confidences[curr_idx+j, token_label] + confidences[curr_idx+j, token_label + other_label]
            
            else: # Can also check the exact correctness
                
                if j == 0:
                    # Add the confidence of being the B- label for the first token
                    res += confidences[curr_idx+j, token_label + other_label]
                
                else:   
                    # Add the confidence of being the I- label for the rest of the tokens
                    res += confidences[curr_idx+j, token_label + other_label]
                

        # Take the average per token probability for this masked sequence
        res /= mask_len

        return res

    

    def calculate_confidence_for_masked_tokens(self, confidences: np.ndarray, i_guess: int) -> np.ndarray:
        """
        Calculate the confidence scores for the masked tokens. Depending on the input, either runs a completely correct document,
        a slice of the correct document (expanding window), or the current guesses.

        inputs: 
            confidences: an array with all of the confidences of the tokens in the document
            i_guess: which guess we're currently doing, or -1 if the true confidences are to be calculated 

        outputs: 
            res: An array containing the confidences for each mask
        """
     
        calculate_true_conf =  i_guess == - 1

        
        if calculate_true_conf:
            # If we are calculating the true confidences, we need to use the true input ids
            
            mask_idxs = self.indices_of_target_masks
            mask_lengths = self.len_of_target_masks
            text_doc = self.target_example
            res = np.zeros(len(mask_idxs)) # Will be a vector containing the confidences for each mask

        elif self.expanding_window : 
            # Slightly different calculation in the centered single approach 
            
            current_start = int(self.EW_centered_single_current_token_idx)
            
            if self.EW_single_entity_replace_target_mask_bool: # If we've replaced the mask with another entity, use the new len instead
            
                current_end = int(current_start + self.EW_single_entity_replaced_mask_len)
            
            else:
            
                current_end = int(current_start + self.EW_single_entity_org_mask_len)

            res = self.calculate_confidence_centered(confidences[current_start: current_end])
            return res
            
        else:
            # Calculating over the guessed tokens
       
            mask_idxs = self.adjusted_indices_of_target_masks
            n_guessed_tokens = len(self.guessed_tokens[i_guess])
            mask_lengths = [len(tokens) for tokens in self.guessed_tokens[i_guess]]
            text_doc = self.trimmed_target_example
            res = np.zeros(len(self.guessed_tokens[i_guess]))

        # For all of the masks
        for i, idx in enumerate(mask_idxs):
            
            if self.expanding_window and (not self.EW_centered_single) and (not calculate_true_conf): 
                
                # If we're outside of the expanding window, break
                if idx > self.EW_current_size:
                    break
          
            
            
            token_label = text_doc['labels'][idx] # True labels
            
            if token_label % 2 == 0: # If the token label is even, the other part of the pair is token_label - 1, otherwise + 1
                other_label = -1
                assert not self.exact_matching_confidence, 'Mask starts with I-label, cannot do exact matching confidence if the labeling is incoherent with the label scheme.' 

            else: 
                other_label = 1
            
            sum_prev_token_length = 0

            # Sum the token probabilities of the entire length of the masked sequence, for both B- and I- labels
            
            # For all of the tokens in the masks
            for j in range(mask_lengths[i]):
                
                assert idx + j < confidences.shape[0], f"Index {idx+j} is out of bounds for dimension 0 with size {confidences.shape[0]}"
                assert token_label < confidences.shape[1], f"Token label {token_label} is out of bounds for dimension 1 with size {confidences.shape[1]}"
                
            
                # Add the probabilities for this masked sequence, for both B- and I- labels
                
                # Can also check the exact correctness
                if self.exact_matching_confidence:

                    if j == 0:
                        # Add the confidence of being the main token label
                        res[i] += confidences[ idx + j + sum_prev_token_length, token_label ]
                    else:   
                        # Add the confidence of being the paired for the rest of the tokens
                        if 0 <= token_label + other_label < confidences.shape[1]:
                            res[i] += confidences[ idx + j + sum_prev_token_length, token_label + other_label ]
                else: 

                    res[i] += confidences[ idx + j + sum_prev_token_length, token_label ] 
                    
                    if 0 <= token_label + other_label < confidences.shape[1]:
                        res[i] += confidences[ idx + j + sum_prev_token_length, token_label + other_label ]
            
            res[i] /= mask_lengths[i] # Divide the current confidence by the number of tokens in the mask


            if not calculate_true_conf:            
                sum_prev_token_length += mask_lengths[i] # Keep track of the number of tokens seen, for EW
            
            if self.expanding_window and (not calculate_true_conf):
                # For the right side expanding window 
                
                if not self.EW_centered_single:
                
                    if idx + sum_prev_token_length > self.EW_current_size: #If we're outside of the current window, stop and return
                
                        return res[:i+1]

        return res


      
        

    def get_doc_from_selections(self, selection_per_PII):
        """
        Combine the document with the guesses.

        The function takes the best candidates for each PII and combines them with the original document.
        The result is a tensor with the same length as the original document, but with the PII replaced by the best candidates.

        selection_per_PII: list
            A list of the candidate to select for each PII position.

        returns:
            best_example: tensor
                The document formed by the best guesss for each PII.
        """

        selected_doc, selected_tokens = self.fill_blanks(indices_of_masks=self.adjusted_indices_of_target_masks,
                                        action_list=selection_per_PII)

        return selected_doc, selected_tokens
        

    def extraction_attack(self, target_doc_idx: int) -> CombinedMetricResult:
        """
        Run the extraction attack on the given documents.
        """
        self.prepare_attack(target_doc_idx)
        self.run_attack()
        best_candidate = self.fetch_best_candidates()
        #return self.calculate_metrics()

    def fetch_best_candidates(self) -> list:
        """
        Fetch the best candidates from the confidence scores.
        """

        # Not sure if good enough. 
        best_candidates = np.argmax(self.confidence_scores, axis=0)
        return best_candidates



    def trim_example(self):
        """
        
        Trim the training example to remove sensitive informaiton. Creates the blank document which is filled in later on during
        the attack. 
        
        For example, if a PII is multiple tokens long, they are merged, as:
        " ... his name is [B-PERSON (Firstname)] [I-PERSON (Lastname)] "  -> becomes ->  "... his name is [B-PERSON]" .

   
        """

        text_labels = self.target_example['labels']
        att_mask = self.target_example["attention_masks"]
        entities = self.target_example['entity_types']
        indices_to_keep = []
        indices_of_masks = []
        length_of_masks = []

        current_mask_tracker = 0            # Keeps track of if we're inside of a mask or not
        n_masks_in_doc = 0
        len_of_current_mask = 0
        prev_label = 0                      # The label of the previous token, used to check if the mask continues
        n_mask_hops = 0                     # The number of times a mask is broken in two due to a single unlabeled token in the middle.

        self.MR_target_idxs = []            # Multi run target indices
        EW_indices = []                     # The indices of the EW entity type, which can be used for selecting a target entity (if random)
      

        for iToken in range(len(text_labels)): # For all of the tokens in the document
            stop_looking = self.limit_masks and (n_masks_in_doc >= self.n_mask_limit) and (len_of_current_mask == 0)
            
            if iToken > 0 and att_mask[iToken] == 0: 
                # We've reached the padding, stop
                break

            if text_labels[iToken] > 0 and (not stop_looking): # If are at a mask, and we're not above the limit of number of masks

                if current_mask_tracker == 0:   # If the mask is starting right now

                    current_mask_tracker = text_labels[iToken]

                    if current_mask_tracker == 0:       # Mask starts with an I- label, not right according to the labeling scheme but kept for now.
                        logger.info( f"Mask starting with I- label at index {iToken} in the document {self.target_doc_idx}." )
                        current_mask_tracker-=1         # Adjust the tracker 

                    indices_to_keep.append(iToken)      # Keep this index in the trimmed document
                    indices_of_masks.append(iToken)     # Save this as a masked index
                    
                    prev_label = current_mask_tracker   # Keep track of the previous label seen ( not used right now )

                    n_masks_in_doc += 1                 # Increase count of seen masks
                    len_of_current_mask = 0             # Mask is starting now, reset length counter

                    if self.multi_run_averaging: # Multi run info
                        if entities[iToken] == self.MR_entity:
                            self.MR_target_idxs.append(iToken)
                    
                    if self.expanding_window: # Expanding window info
                        if entities[iToken] == self.EW_entity_type_int:
                            EW_indices.append(iToken)


                elif text_labels[iToken] != current_mask_tracker + 1:       # If we've reached a new mask directly after a previous mask
                    
                    length_of_masks.append(len_of_current_mask)             # Save the length of the previous mask
                    len_of_current_mask = 0                                 # Reset it for the current one

                    indices_to_keep.append(iToken)                          # Keep this index in trimmed document
                    indices_of_masks.append(iToken)                         # Save this index as a masked index

                    current_mask_tracker = text_labels[iToken]              # Overwrite the current mask tracker

                    n_masks_in_doc += 1                                     # Increase count of seen masks

                    if current_mask_tracker % 2 == 0:       # Mask starts with an I- label, not right according to the labeling scheme but kept for now.
                        logger.info(f"Mask starting with I- label at index {iToken} in the document {self.target_doc_idx}.")
                        current_mask_tracker-=1             # Adjust the tracker


                    if self.multi_run_averaging: # Multi run info
                        if entities[iToken] == self.MR_entity:
                            self.MR_target_idxs.append(iToken)

                    if self.expanding_window: # Expanding window info
                        if entities[iToken] == self.EW_entity_type_int:
                            EW_indices.append(iToken)

                len_of_current_mask += 1 # Increase the count of the current mask
            
            # If we're at a non-mask token but just afterwards it continues with the same mask. Dont use this for now, but should think about it. 
            elif ( text_labels[iToken] == 0) and ( iToken + 1 < len(text_labels) ) and (text_labels[iToken + 1] == prev_label+1) and False: 
                len_of_current_mask += 1

            else: # We're outside of a mask

                if ( text_labels[iToken] == 0) and ( 0 < iToken < len(text_labels) -1 ) and ( text_labels[iToken + 1] == text_labels[iToken - 1]) : 
                    n_mask_hops += 1 # There's a single skip in the labels
                
                if current_mask_tracker != 0: # If we were just in a mask
                    length_of_masks.append(len_of_current_mask) # Save lenght of previous seen mask
    
                current_mask_tracker = 0
                len_of_current_mask = 0
                indices_to_keep.append(iToken)

        logger.info(f' Target document trimmed. {n_mask_hops} "hops" were found in the document. ')
        
        # set the trimmed target example 
        self.trimmed_target_example = { 'input_ids' : self.target_example["input_ids"][indices_to_keep],  # The ids of the tokens
                            'labels' :text_labels[indices_to_keep],
                            'attention_masks' : att_mask[indices_to_keep],
                            'identifier_types' : [self.target_example["identifier_types"][i] for i in indices_to_keep],
                            #'offsets' : [training_ex["offsets"][i] for i in indices_to_keep],
                            'entity_types' : [self.target_example["entity_types"][i] for i in indices_to_keep] }
            
        self.n_masks_in_target = n_masks_in_doc             # Set the number of masks in document
        self.indices_of_target_masks = indices_of_masks     # Set the indices of the masks
        self.len_of_target_masks = length_of_masks          # Set the length of the masks
        
        if self.multi_run_averaging:
            self.MR_target_n = len(self.MR_target_idxs)
            self.MR_remaining_n = n_masks_in_doc - self.MR_target_n # How many remaining masks there are

        
        
        if self.expanding_window:

            # If we're going to investigate a single mask
            if self.EW_single_entity_bool:     
                
                # If we're to use a random index from the doc, overwrite the currently selected one by a random one
                if self.EW_single_entity_random_mask_bool:
                    
                    self.EW_single_entity_actual_mask_idx = np.random.choice(EW_indices)




    def fill_blanks(self, indices_of_masks = None, action_list = None):
        """
        Fill in the blanks in the trimmed text document.
        """
       
        # Initiate some stuff
        chosen_tokens = []
        max_token_len = MODEL_MAX_LEN  # model maximum token length
        initial_token_length = len(self.trimmed_target_example['input_ids'])

        filled_doc_ids = []
        filled_doc_labels = []
        filled_doc_offsets = []
        filled_doc_entity_types = []
        filled_doc_identifier_types = []
        prev_idx = 0 # Keeps track of tokens from the trimmed doc to add
        
        # Single investigation: A specific idx is frozen as the correct one, and an increasing number of 
        # the remaining tokens are filled by the correct value as time goes on. In the end all tokens should be correct.
        if self.single_investigation:
            q = self.attack_attempt / self.n_evaluations 
            n_to_fix = min( self.n_masks_in_target, int( q * self.n_masks_in_target ) )
            current_forced_correct = range( n_to_fix )

            # Save the nr of fixed tokens for later display
            self.set_n_correct_guesses(n_to_fix, self.attack_attempt)
        
        if self.multi_run_averaging:
            # sample a fraction of the samples of the remaining ones to be correct, leave rest true
            n_forced_correct = int(self.MR_correct_fractions[self.attack_attempt]*self.MR_remaining_n)
            if self.attack_attempt >= self.n_evaluations-1:
                n_forced_correct = self.MR_remaining_n

            true_fraction_correct = n_forced_correct/self.MR_remaining_n
            self.MR_actual_correct_fractions.append(true_fraction_correct) # Solves issue of different percentages giving same nr of forced correct
            indices_to_set_correct = np.random.choice(range(self.MR_remaining_n), n_forced_correct) # TODO: combine with EW experiment, need to make sure they are not larger than the indices in the current window 
            remaining_indices = np.array(list(set(self.indices_of_target_masks) - set(self.MR_target_idxs)))
            
            if len(indices_to_set_correct) != 0:
                indices_to_set_correct = remaining_indices[indices_to_set_correct]


        for iMask, token_idx in enumerate(sorted(indices_of_masks, reverse=False)): # For each mask
            
            
            if self.expanding_window: 
                if token_idx > self.EW_current_size: # If we're outside of the expanding window, break the loop. 
                    break

            # Integer labels for the Beginning- and In-tokens
            b_label = self.trimmed_target_example['labels'][token_idx]
            
            if b_label % 2 == 0: 
                b_label -= 1
            
            i_label = b_label + 1
            
            if action_list: # If there's an action list, we will sample accordingly
                entity_to_sample = action_list[iMask] 
            else: 
                entity_to_sample = -1 # Otherwise, random selection

            if self.single_investigation: # If we're investigating a single specific mask 
                if (iMask in current_forced_correct ) or (iMask == self.single_mask_idx) : # If it's the specified mask, or in the forced true ones, set the action to be the true one
                    entity_to_sample = self.pool.true_idxs[iMask]
            
            
            
            # Set which pool to use
            if self.global_attack_pool_bool:
                local_pool_id = -1
            else: 
                local_pool_id = iMask

            # Set entity type of the mask to sample
            entity_type = self.trimmed_target_example['entity_types'][token_idx].item() 
                        
            # Multi-run averaging
            if self.multi_run_averaging: # Note that this overrides all of the other strategies.
                # If we're at the specified entity type, we input the correct value
                if entity_type == self.MR_entity: 
                    entity_to_sample = self.pool.true_idxs[iMask]
                
                # Otherwise, with prob self.multi_run_random_fraction, set the remaining masks as either true or random.
                else: 
                    if iMask in indices_to_set_correct:
                        entity_to_sample = self.pool.true_idxs[iMask]
                    else: 
                        entity_to_sample = -1

            # Expanding window
            if self.expanding_window:

                if self.EW_single_entity_bool:
                
                    if self.EW_single_entity_actual_mask_idx == iMask:
                
                        entity_to_sample = self.pool.true_idxs[iMask]
                
                    else:
                
                        if self.EW_single_entity_set_rest_correct_bool: 
                
                            entity_to_sample = self.pool.true_idxs[iMask]
                
                        else: 
                            
                            entity_to_sample = -1
                
                else: 
                
                    if self.EW_entity_type_int == entity_type:
                
                        entity_to_sample = self.pool.true_idxs[iMask]
                
                    else:
                
                        if self.EW_single_entity_set_rest_correct_bool: 
                
                            entity_to_sample = self.pool.true_idxs[iMask]
                
                        else: 
                            
                            entity_to_sample = -1


            # Samples a piece of text to inpaint into the masked position
            tokenized_text = self.pool.sample_from_candidates(entity_type = entity_type, 
                                                              entity_id = entity_to_sample, 
                                                              local_pool_id = local_pool_id )
            
            tokenized_len = len(tokenized_text) 
            chosen_tokens.append(tokenized_text)
            
            # Add the unmasked segment between the two most recent tokens and the new tokenized text to the resulting document
            filled_doc_ids = filled_doc_ids + self.trimmed_target_example["input_ids"][prev_idx : token_idx].tolist() + tokenized_text.tolist()
            filled_doc_labels = filled_doc_labels + self.trimmed_target_example["labels"][prev_idx : token_idx].tolist() + [b_label.item()] + [i_label.item() for j in range(tokenized_len - 1)]   
            filled_doc_entity_types = filled_doc_entity_types + self.trimmed_target_example["entity_types"][prev_idx : token_idx] + tokenized_len*[self.trimmed_target_example['entity_types'][token_idx]]
            filled_doc_identifier_types = filled_doc_identifier_types + self.trimmed_target_example["identifier_types"][prev_idx : token_idx] + tokenized_len*[self.trimmed_target_example['identifier_types'][token_idx]]

            prev_idx = token_idx +1

        if self.expanding_window: 
            # In case we're using expanding window, slice off the end. 
            filled_doc_ids = filled_doc_ids + self.trimmed_target_example["input_ids"][prev_idx :min(initial_token_length, self.EW_current_end)].tolist() 
            filled_doc_labels = filled_doc_labels + self.trimmed_target_example["labels"][prev_idx : min(initial_token_length, self.EW_current_end)].tolist() 
            filled_doc_entity_types = filled_doc_entity_types + self.trimmed_target_example["entity_types"][prev_idx : min(initial_token_length, self.EW_current_end)]
            filled_doc_identifier_types = filled_doc_identifier_types + self.trimmed_target_example["identifier_types"][prev_idx :min(initial_token_length, self.EW_current_end)]
            filled_doc_attention_masks = [1 for i in range(len(filled_doc_ids))]
        
        else:
            # Add the final unmasked segment to the filled document
            filled_doc_ids = filled_doc_ids + self.trimmed_target_example["input_ids"][prev_idx : initial_token_length].tolist() 
            filled_doc_labels = filled_doc_labels + self.trimmed_target_example["labels"][prev_idx : initial_token_length].tolist() 
            filled_doc_entity_types = filled_doc_entity_types + self.trimmed_target_example["entity_types"][prev_idx : initial_token_length]
            filled_doc_identifier_types = filled_doc_identifier_types + self.trimmed_target_example["identifier_types"][prev_idx : initial_token_length] 
            filled_doc_attention_masks = [1 for i in range(len(filled_doc_ids))]

        
        if filled_doc_ids[-1] != 2: # If the last symbol is not the EOS token, add it and extend the other lists
            filled_doc_ids = filled_doc_ids + [2] 
            filled_doc_labels = filled_doc_labels + [-1]
            filled_doc_entity_types = filled_doc_entity_types + [-1]
            filled_doc_identifier_types = filled_doc_identifier_types +  [-1]
            filled_doc_attention_masks =  filled_doc_attention_masks + [1]
        
        # Add the padding to the filled document. If it's negative, we've created a too long sequence somehow. 
        padding_to_add = max_token_len - len(filled_doc_ids)
        
        assert padding_to_add >= 0, f'The token list is too long. '  




        # Prolly need to make sure that the padding is done correctly, tensors and on device
        filled_text_example ={ 'input_ids' : tensor(filled_doc_ids + padding_to_add*[self.tokenizer.pad_token_id], dtype = int),
                            'labels' : tensor(filled_doc_labels + padding_to_add* [-1], dtype = int),
                            'attention_masks' : tensor(filled_doc_attention_masks + padding_to_add*[0], dtype = int),
                            'identifier_types' : tensor(filled_doc_identifier_types + padding_to_add*[-1], dtype = int),
                            #'offsets' : text_offsets[:pos] + padding_to_add*[-1],
                            'entity_types' : tensor(filled_doc_entity_types + padding_to_add*[-1], dtype = int)}
        
        self.attack_attempt += 1

        return filled_text_example, chosen_tokens
    
    def set_n_correct_guesses(self, number_of_correct_guesses, i_attempt):
        self.n_corr_per_guess[i_attempt] = number_of_correct_guesses
    

    # Abstract methods
    def description(self) -> dict:
        """Return a description of the attack."""
        title_str = "Mastermind Attack"
        reference_str = " - "
        summary_str = "Reconstruction attacks based on token level confidences and access to population pool."
        detailed_str = " - "
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }
    



# ------------------------- SAVE & ANALYSIS FUNCTIONS -------------------------
    
    def gather_statistics(self):
        
        """ Gather statistics about the confidence of the masks withouth any context"""
    
        
        n_total_docs = self.population_length
        true_probs_of_all = []
        entity_wise_probs = {}

        first_mask_records = []

        target_model = self.target_model.to(self.device)
        target_model.eval()
        n_diff_docs = 0
        n_seen_masks = 0
        true_probs_of_all_dict = {}
        for iEntityType in range(8):
            true_probs_of_all_dict[iEntityType] = []

        with no_grad():

            for iDoc in tqdm(range(n_total_docs)):
                
                self.target_doc_idx = iDoc
                self.target_example = self.handler.population[self.target_doc_idx]
            
                if (self.target_example['labels'] > 16).any(): 
                    print('Out of reach labels found in doc ', iDoc)
                    continue
               
                self.trim_example()
                
                len_of_target_example = len(self.target_example['input_ids'])
                self.true_input_ids = cat((self.target_example['input_ids'], ones(MODEL_MAX_LEN  - len_of_target_example, dtype = int)))
                self.true_entity_types = cat( (tensor(self.target_example['entity_types'], dtype = int), -1*ones(MODEL_MAX_LEN  - len_of_target_example, dtype = int))) # Add -1 for the padding at the end, so that the shape is correct


                # A way of making sure we don't investigate the same doument multiple times, REMEMBER however that there are multiple annotations per document
                first_mask = self.target_example['input_ids'][self.indices_of_target_masks[0]:self.indices_of_target_masks[0] + self.len_of_target_masks[0]]
                first_mask_hash =  first_mask.cpu().numpy().tobytes() # Convert to bytes for hashing
                if first_mask_hash in first_mask_records: 
                    continue
                
                else:

                    first_mask_records.append(first_mask_hash)
                    
                    
                    # TODO save the entity types of the true probabilities. 
                    current_true_entity_types = np.array(self.target_example['entity_types'])[self.indices_of_target_masks]
                    true_probs = self.get_true_confidences()
                    for entity_type, true_conf in zip(current_true_entity_types, true_probs):
                        true_probs_of_all_dict[entity_type].append(true_conf)

                    true_probs_of_all.append(self.get_true_confidences())
                    # We've reached a new document
                    n_diff_docs += 1
                    n_seen_masks += self.n_masks_in_target
             
                    for iMask in range(self.n_masks_in_target): # For each mask in the document, extract the context-free confidence
                            
                        # Retrieve info related to the mask
                        current_mask_token_idx = self.indices_of_target_masks[iMask]
                        len_of_current_mask = self.len_of_target_masks[iMask]
                        current_entity_type = self.true_entity_types[current_mask_token_idx].item()
                        current_label = self.target_example['labels'][current_mask_token_idx].item()

                        padding_to_add = MODEL_MAX_LEN  - (len_of_current_mask + 2)

                        # Create a masked sequence which is only the mask with padding 
                        masked_seq = self.target_example['input_ids'][self.indices_of_target_masks[iMask]:self.indices_of_target_masks[iMask] + self.len_of_target_masks[iMask]].clone().detach()
                        input_ids = cat( [ tensor([0], dtype = int), masked_seq, tensor([2], dtype = int), ones(padding_to_add, dtype = int) ] )
                        attention_mask = cat([ones(MODEL_MAX_LEN  - padding_to_add, dtype = int), zeros(padding_to_add, dtype = int) ])

                        # Extract the confidence
                        logits_from_model = self.target_model(input_ids = input_ids.unsqueeze(0).to(self.device), attention_mask = attention_mask.unsqueeze(0).to(self.device))
                        probs_from_model = softmax_logits(logits_from_model[0][1:1+len_of_current_mask].cpu().numpy())
                        
                        if current_label % 2 == 0: 
                            other_label = -1
                        else: 
                            other_label = 1

                        if len(probs_from_model.shape) < 2: # If the model doesn't output probabilities for both B- and I- labels, skip this mask
                            
                            current_confidence = np.sum( probs_from_model[current_label] + probs_from_model[current_label + other_label]) / len_of_current_mask # Average confidence of the mask
                        
                        else: 
                            current_confidence = np.sum( probs_from_model[:, current_label] + probs_from_model[:, current_label + other_label]) / len_of_current_mask # Average confidence of the mask
                    

                        if current_entity_type not in entity_wise_probs.keys():
                            entity_wise_probs[current_entity_type] = [] 

                        entity_wise_probs[current_entity_type].append(current_confidence)
        
        # Save results

        np.save(self.path_to_output + '/true_probs_of_all_dict.npy', true_probs_of_all_dict)
        np.save(self.path_to_output + '/entity_wise_probs.npy', entity_wise_probs)

        true_probs_of_all = [prob for doc in true_probs_of_all for prob in doc] # Flattening of a nested list. 
        print('n_seen_masks:', n_seen_masks)
        print('n_seen_docs:', n_diff_docs)
        
        # Make histogram of the true confidences
        plt.figure()
        plt.hist(true_probs_of_all, bins = 100, density = True, alpha = 0.5, label = 'Confidences')
        title_string1 = f'True Confidences of the masks in the full documents. \n{n_seen_masks} masks, taken from {n_diff_docs} documents.'
        plt.title(title_string1)
        plt.grid()     
        plt.legend()
        plt.savefig( self.figpath + 'true_probs_histogram.svg')
        
        # Make histogram of confidences of context free masks
        
        entity_labels, entity_hist = zip(*entity_wise_probs.items())
        entity_labels = [self.entity_int_to_string[el] for el in entity_labels]


        plt.figure()
        plt.hist([e for entity_conf in entity_hist for e in entity_conf], bins = 100, density = True, alpha = 0.5, label = 'Confidences')
        plt.grid()        
        plt.legend()
        title_string2 = f'Context-free confidences of the masks. \n {n_seen_masks} masks, taken from {n_diff_docs} documents.'
        plt.title(title_string2)
        plt.savefig( self.figpath + 'context_free_confidence_histogram.svg')

        
        plt.figure()
        plt.hist(entity_hist, bins = 100, density = True, stacked = True, alpha=0.5, label = entity_labels)
        title_string3 = f'Context-free confidences of the masks by entity type. \n {n_seen_masks} masks, taken from {n_diff_docs} documents.'
        plt.title(title_string3)
        plt.legend()
        plt.grid() 
        plt.savefig( self.figpath + 'context_free_confidence_entity_wise_histogram.svg')

        filtered_pairs = [( hist, label) for label, hist in zip(entity_labels, entity_hist) if label != "DATETIME"]
        entity_hist_no_datetime, entity_labels_no_datetime = zip(*filtered_pairs)

        plt.figure()
        plt.hist(entity_hist_no_datetime, bins = 100, density = True, stacked = True, alpha=0.5, label = entity_labels_no_datetime)
        title_string4 = f'Context-free confidences of the masks by entity type, without DATETIME. \n {n_seen_masks} masks, taken from {n_diff_docs} documents.'
        plt.title(title_string4)
        plt.legend()
        plt.grid() 
        plt.savefig( self.figpath + 'context_free_confidence_entity_wise_histogram_no_datetime.svg')


        logger.info(f'Gathered statistics done. Investigated {n_diff_docs} different documets out of {self.population_length} in total (different annotations ignored).')
    
            




    def generate_results(self):
        """ 
        Generate the results of the attack, including printing and plotting.
        Should prob be set in config dict, or in the init.
        """

        print_res = False
        plot_res = True
        save_res = False
        

        self.print_results(print_res)
        self.plot_results(plot_res)
        self.save_results(save_res)

 
        if self.expanding_window:
        
            logger.info('Plotted Expanding Window Investigation Results.')

        if self.single_investigation:
            
            logger.info('Plotted Single Mask Investigation Results.')
        
        # Save the results
        self.save_results()
    
    def print_results(self, print_res = False):

        if print_res: 
            self.print_summary()
            self.print_info_per_pool()
            logger.info('Written results generated')

    def plot_results(self, plot_res = False):
        
        if plot_res:
            
            

            if self.expanding_window:
                plot_expanding_window_result(self)
        
            elif self.single_investigation:
                plot_single_mask_results(self)
            else:
                plot_info_per_pool(self)
                plot_histograms(self)

            logger.info('Plotted results generated')

    def save_results(self, save_res = False):
       
        if save_res:        
            
            if self.single_investigation:
            
                np.save(self.path_to_output + "/single_investigation_conf.npy", self.single_res)
        
            else:

                np.save(self.path_to_output + "/guessed_tokens.npy", self.guessed_tokens)
                np.save(self.path_to_output + "/actions_taken.npy", self.actions_taken)
                np.save(self.path_to_output + "/observations.npy", self.observed_rewards)
                
            if self.expanding_window:
        
                np.save(self.path_to_output + '/EW_size_history.npy', self.EW_size_history)
            


    def save_multi_run_results(self):
        """ Save the results of the multi-run attack, saves a file per doc after attack run on that doc."""
        
        dct = {
            'fractions':    self.MR_actual_correct_fractions,
            'observed_rewards': self.observed_rewards,
            'entity_types': self.true_entity_types,
            'target_entity_type': self.MR_entity,
            'n_masks_in_target':  self.n_masks_in_target,
            'n_masks_of_target_type':  self.MR_target_n,
            'doc_id':self.multi_run_target_doc_idx
            }
        
        np.save(self.path_to_output + "/mra_res_dct_run_" + str(self.MR_counter), dct)

    def print_summary(self) -> None:
        """
        First (optionally) prints a brief summary to the terminal, then fills the original text with the best candidate for each PII 
        and prints it into the file best_text.txt.
        """

        # Brief Summary
        print_to_terminal = False
        print_res_per_PII = True

        correct_guesses = 0
        correct_indices = []
        almost_correct_guesses = 0
        top_count_init = 5
        
        top_count = top_count_init*ones(self.n_masks_in_target, dtype = int)
        top_count_per_action = 0
        for i in range(self.n_masks_in_target): # For each masked position
            
            if self.attack_strategy == "bandit" and False:
                rewards_per_action = self.extraction_game.players[i].get_policy()
            else:
                rewards_per_action = [self.confidence_scores[j][i] for j in range(self.n_evaluations)]
            # print the top 5 guesses and the true word
           
            top_count[i] = min(top_count[i], len(rewards_per_action)-1) 


            # NOTE: These actions taken for the top rewards are not unique, i.e. the top actions can all be the same
            top_five_guesses = np.argsort(rewards_per_action)[-top_count[i]:][::-1]
            top_five_ids = [self.actions_taken[j][i] for j in top_five_guesses]
            
            individual_action_reward = False
            if individual_action_reward: 
                    
                sorted_rewards = np.argsort(rewards_per_action)[::-1]
                found_actions = []
                for j in sorted_rewards:
                    if self.actions_taken[j][i] not in found_actions:
                        found_actions.append(self.actions_taken[j][i])
                    if len(found_actions) == 5:
                        break
                if 0 in found_actions:
                    top_count_per_action +=1
                print("Actions taken, sorted by maximum reward per action: ", found_actions)
            
            #TODO label to int
            true_entity_type_int = self.true_entity_types[i].item()
            true_entity_type_string = self.entity_int_to_string[true_entity_type_int]

            correct_tokens = self.true_input_ids[i]      
            if self.pool.local_pool:
                best_found_tokens = self.pool.local_pool[i][top_five_ids[0]]  
            else: 
                best_found_tokens = self.pool.attack_pool[true_entity_type_int][top_five_ids[0]] 
            true_text = self.tokenizer.decode(correct_tokens)
            best_guess = self.tokenizer.decode(best_found_tokens)
            
            
            if print_to_terminal and print_res_per_PII: 
                print(f"True Entity: {true_entity_type_string}, True text: {true_text}, Best Guess: {best_guess}")
            
            just_found_correct = False
            taken_action = top_five_ids[0]
            
            correct_action = self.pool.true_idxs[i]
            #print(f"True Action = ", correct_action, "Taken Action = ", taken_action)
            if (taken_action == correct_action) or (true_text.strip().lower() == best_guess.strip().lower()) :
                correct_guesses += 1
                correct_indices.append(i)
                just_found_correct = True
                self.binary_correct_guesses[i] = 1
            
            for j in range(top_count[i]):
                if self.pool.local_pool: 
                    almost_best_guess = self.tokenizer.decode(self.pool.local_pool[i][top_five_ids[j]])
                else: 
                    almost_best_guess = self.tokenizer.decode(self.pool.attack_pool[true_entity_type_int][top_five_ids[j]])
                taken_action = top_five_ids[j]
                if true_text.strip().lower() == almost_best_guess.strip().lower() or (taken_action == correct_action):
                    almost_correct_guesses += 1
                    break
            if print_to_terminal and print_res_per_PII:
                if just_found_correct:
                    print("Correct guess")

        logger.info(f'Number of completely correctly identified PII after {self.n_evaluations} evaluations are {correct_guesses}/{self.n_masks_in_target} = {100*correct_guesses/self.n_masks_in_target:.2f}%.')
        logger.info(f'Correct indices are {correct_indices}')
        logger.info(f'Number of *almost* correctly identified PII after {self.n_evaluations} evaluations are {almost_correct_guesses}/{self.n_masks_in_target} = {100*almost_correct_guesses/self.n_masks_in_target:.2f}%.')
        logger.info(f"Number of the times the correct actions were in the 5 best actions is {top_count_per_action}/{self.n_masks_in_target} = {100*top_count_per_action/self.n_masks_in_target:.2f}%.")
        self.final_n_correct = correct_guesses
        self.final_n_almost_correct = almost_correct_guesses
        
        # Print the original text to best_estimate_text.txt, and then the best candidate option in the same doc
        with open(self.path_to_output + "/best_estimate_text.txt", "w") as f:
            
            true_ids = self.target_features['input_ids']
            true_text = ''
            mem = 0

            for i, idx in enumerate(self.indices_of_target_masks):
            
                if self.attack_strategy == "bandit":
            
                    best_action = np.argmax(self.extraction_game.players[i].get_policy())
            
                    if self.pool.local_pool: 
            
                        best_candidates = self.pool.local_pool[i][best_action]
            
                    else:
            
                        best_candidates = self.pool.attack_pool[self.true_entity_types[i].item()][best_action]
            
                else: 
            
                    best_action = np.argmax(self.confidence_scores[:, i])
            
                    if self.pool.local_pool: 
            
                        best_candidates = self.pool.local_pool[i][best_action]
            
                    else:
            
                        best_candidates = self.pool.attack_pool[self.true_entity_types[i].item()][best_action]
                
                # Get the best candidate for the current mask
                true_text += self.tokenizer.decode(true_ids[mem:idx])
                true_text += " [ ESTIMATE: '"
                true_text += self.tokenizer.decode(best_candidates) 
                true_candidate = self.tokenizer.decode(true_ids[idx: idx+ self.len_of_target_masks[i]])
                true_text += f"' / TRUE: '{true_candidate}' ]"
                mem = idx + self.len_of_target_masks[i]
            
            # Add the remaining text
      
            final_ids = true_ids[mem:len(true_ids)-1]
            final_text = final_ids[np.where(final_ids != 1)[0]]     
            true_text += self.tokenizer.decode(final_text)
       
            f.write("\nBest estimated text: \n")
            f.write(true_text)
            f.close()
        
        logger.info("Text written to best_estimate_text.txt")


    def print_info_per_pool(self):
        """
        For each pool, print some info such as: 
        Size of pool;
        entries in text per pool, 
        Which entries were tried in the evaluations
        What scores were given during the different evaluations
        """

        n_mask_in_summary_limit = 7 # Only print the first 7 PII in the summary, for readability
        
        with open("./pool_info.txt", "w") as output_file:
            
            output_file.write("Pool info:\n")

            for key in self.pool.attack_pool.keys():
                
                output_file.write(f"Pool size for key {key} ({self.entity_int_to_string[key]}): {len(self.pool.attack_pool[key])}\n")
                output_file.write(f"Entries in pool:\n") 
                
                for i, text in enumerate(self.pool.attack_pool[key]):
                
                    output_file.write(f" {i}: '{self.tokenizer.decode(text)}'\n")
                    

            for iMask, idxMask in enumerate(self.indices_of_target_masks):
                
                curr_entity_type = self.true_entity_types[iMask].item()
                corr_id = self.pool.true_idxs[iMask]

                output_file.write(f"PII nr {iMask}")
                output_file.write(f" True entity idx (action) in pool: {corr_id} \n")
                
                
                corr_id_conf = []
                incorr_id_conf = []
                dct = {}
                
                max_val = -1
                max_val_idx = -1
                
                for iAttempt in range(self.n_evaluations):

                    selected_action = self.actions_taken[iAttempt][iMask]
                    curr_conf = self.confidence_scores[iAttempt][iMask]
                    
                    if curr_conf > max_val: 
                        max_val = curr_conf
                        max_val_idx = selected_action
                    
                    if selected_action not in dct:
                        dct[selected_action] = []
                    
                    dct[selected_action].append(curr_conf)
                    
                    if selected_action == corr_id:

                        corr_id_conf.append(curr_conf)
                    
                    else:
                        incorr_id_conf.append(curr_conf)

                    output_file.write(f" Action {selected_action}{' (Correct)' if selected_action == corr_id else ''}, score:  {curr_conf}\n")
                    
                
                if iMask>=n_mask_in_summary_limit: # Only print the first 7 PII ( arbitrary limit for readability)
                
                    break 

            output_file.write('\n')
