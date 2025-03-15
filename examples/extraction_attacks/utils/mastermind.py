"""Implementation of the Mastermind extraction attack."""
import numpy as np

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
from torch import device, optim, cuda, no_grad, save, sigmoid, Tensor, zeros, tensor, ones, long, argmax
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from examples.extraction_attacks.utils.src.games import ExtractionGame
from tqdm import tqdm  


class AttackMM(AbstractMIA):
    def __init__(self: Self, handler: AbstractInputHandler, configs: dict, pool: PIIPool = None) -> None:
        super().__init__(handler)
        self.pool = pool
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.tokenizer = None
        self._configure_attack(configs)



    def _configure_attack(self, configs) -> None:

        # General configs
        self.attack_strategy = configs.get("attack_strategy", "random") # "random" is default
        self.n_evaluations = configs.get("n_filling_attempts", 10) # How many times we try to impute the document with text according to the strategy
        self.attack_pool_type = configs.get("attack_pool_type", "full") # full or extended
        self.attack_pool_size_extension = configs.get("attack_pool_size_extension", 0) # How many extra entities to add to the attack pool, aside from the correct ones
        self.bandit_parameters = configs.get("bandit_parameters", None) # Parameters for the bandit strategy
        
        # Form the pool of the data
        self.data_pool_path = configs.get("data_pool_path", None)
        if self.pool is None:
            if self.data_pool_path is None:
                raise ValueError("No datapool path provided")
            self.pool = PIIPool(self.data_pool_path)
        if self.attack_pool_type == "extended":
            self.pool.extend_pool(self.attack_pool_size_extension)

        # Set the target document to attack TODO: Change this to be more flexible
        self.target_doc_id = configs.get("target_doc_id", 0)

        # Retrieve the features and labels of the target document
        self.target_features, self.target_labels = self.handler.population[self.target_doc_id]
    
        # Trim the documents and extract the number of tokens to fill
        self.trimmed_target_example, self.n_tokens_to_fill, self.target_token_idx, self.len_of_target_masks = self.trim_example(self.target_features, self.target_labels) 
        
        # Set the tokenizer and target model
        self.target_model = self.handler.target_model
        self.tokenizer = self.handler.population.tokenizer

        # Initialize the confidence score arrays
        self.confidence_scores = np.zeros((self.n_evaluations, self.n_tokens_to_fill))
        
        # Initialize the token arrays
        self.tokens_target_doc = [[] for i in range(self.n_evaluations)]

      


    def prepare_attack(self) -> None:
        pass

    def run_attack(self):
        observed_rewards = []
        
        # Random attack
        if self.attack_strategy == "random":
            logger.info("Using random sampling strategy")
            target_model = self.target_model.to(self.device)
            target_model.eval()
        
            for i in range(self.n_evaluations):
            
                attack_doc, token_ids = self.fill_blanks(self.trimmed_target_example, self.n_tokens_to_fill)
                self.tokens_target_doc[i] = token_ids
                
               

                with no_grad():
                    
                    target_feats, target_labels  = self.custom_collate_fn(attack_doc)

                    logits_target = target_model(target_feats)
                    probs_target = softmax_logits(logits_target.cpu().numpy())
                    probs_of_masks = self.calculate_confidence_for_masked_tokens(probs = probs_target, i_guess=i)
                    self.confidence_scores[i] = probs_of_masks

                    


            logger.info("Iterations Completed")
            observed_rewards = self.confidence_scores

        # Bandit Attack TODO: Might need changing if more bandit algorithms are added
        elif self.attack_strategy == "bandit": 
            logger.info("Using Tsallis inf sampling strategy")
            T = self.n_evaluations
            mask_types = self.target_labels[self.target_token_idx]
            extraction_game = ExtractionGame(T = T, 
                                             n_masks = self.n_tokens_to_fill, 
                                             mask_types = mask_types, 
                                             pool = self.pool,
                                             attack_obj = self )
            
            observed_rewards = []
            extraction_game.reset_game()
            
            correct_guesses_over_time = np.zeros(extraction_game.num_players)
            for t in tqdm(range(T)):
                observed_rewards.append(extraction_game.step())
                extraction_game.update_policies(observed_rewards[t], t)
            
            logger.info("Iterations Completed")

            label_set = extraction_game.label_set
            correct_guesses = 0
            with open(f"player_actions.txt", "w") as f:
                # Save actions for each player
                for i in range(extraction_game.num_players):
                    actions = extraction_game.players[i].get_policy()
                    
                    # print the top 5 guesses and the true word
                    top_five_ids = np.argsort(actions)[-5:][::-1]
                    label_of_mask = label_set.ids_to_label[mask_types[i].item()][2:]
                    


                    correct_tokens = self.target_features['input_ids'][self.target_token_idx[i]:self.target_token_idx[i] + self.len_of_target_masks[i]]                    
                    best_found_tokens = self.tokenizer.encode(self.pool.candidate_pool[label_of_mask][top_five_ids[0]])[1:-1] 
                    
                    print(f"True Entity: {label_of_mask}, True text: {self.tokenizer.decode(correct_tokens)}")
                    print(f"Five best guesses and weights in policy for player {i}:")
                    print(f"{[f'{self.pool.candidate_pool[label_of_mask][j]}, {actions[j]}' for j in top_five_ids]}")
                    
                    if len(correct_tokens) == len(best_found_tokens):
                        if correct_tokens == best_found_tokens:
                            correct_guesses += 1
                    #f.write(f"Player {i}: {actions}")

                #print(f"Learned policy of player {i} is: {[f'{action:.6f}' for action in actions]}")
            print(f'Number of completely correct identified tokens after {self.n_evaluations} evaluations are {correct_guesses},\nwhich is an accuracy of {100*correct_guesses/self.n_evaluations:.4f}%.')
      
        size_dict = self.pool.get_pool_sizes()
        for k, v in size_dict.items():
            print(f"Key {k}, length {v}")
        
        
        #self.print_results(self.fetch_best_candidates("member"))
        print("Results printed")
        np.save("./observations.npy", observed_rewards)
    


    def calculate_confidence_for_masked_tokens(self, probs: np.ndarray, i_guess: int) -> np.ndarray:
        """
        Calculate the confidence scores for the masked tokens.
        """
        token_idxs = self.target_token_idx
        token_lengths = [len(tokens) for tokens in self.tokens_target_doc[i_guess]]


        res = np.zeros(len(token_idxs))
        for i, idx in enumerate(token_idxs):
            # Get the correct classes of the masked tokens
            token_label = self.trimmed_target_example['labels'][idx]
            # Sum the token probabilities of the entire length of the masked sequence, for both B- and I- labels
            for j in range(token_lengths[i]):
                assert idx + j < probs.shape[0], f"Index {idx+j} is out of bounds for dimension 0 with size {probs.shape[0]}"
                assert token_label < probs.shape[1], f"Token label {token_label} is out of bounds for dimension 1 with size {probs.shape[1]}"
                assert token_label+1 < probs.shape[1], f"Token label+1 {token_label+1} is out of bounds for dimension 1 with size {probs.shape[1]}"
    
                res[i] += probs[idx+j, token_label] + probs[idx+j, token_label+1]
            # Take the average of the probabilities for this masked sequence, for both B- and I- labels
            res[i] /= token_lengths[i]

        return res

    def print_results(self, best_candidates: list, ) -> None:
        """
        Print the results of the attack.
        """
        # Print the original text to org_text.txt, and then the best candidate option in the same doc
        with open("org_text.txt", "w") as f:
            
            true_text = self.target_features['input_ids'][self.target_features['input_ids'] != self.tokenizer.pad_token_id]
            
            f.write("\nTrue text: \n")
            f.write(self.tokenizer.decode(true_text[1:-1]))
            
            #f.write("\nFinal text: \n")
            #final_text = self.combine_doc_with_guesses(best_candidates)
            #f.write(self.tokenizer.decode(final_text))


            f.write("\n\n\n")
            f.write("Guesses for the masks: \n")
            for i in range(len(best_candidates)):
                f.write(f"True: ")
                f.write(self.tokenizer.decode(self.target_features['input_ids'][self.target_token_idx[i]:self.target_token_idx[i]+self.len_of_target_masks[i]]))
                f.write("\n")
                f.write(f"Guess: ")
                f.write(self.tokenizer.decode(self.tokens_target_doc[best_candidates[i]][i]))
                f.write("\n\n")
        

    def combine_doc_with_guesses(self, best_candidates):
        """
        Combine the document with the guesses.
        """
        full_token_list = []
        text_pointer = 1
        
        for i in range(len(best_candidates) - 1):
                
                # Append the unmasked tokens from the original document
                full_token_list.append(self.trimmed_target_example['input_ids'][text_pointer:self.target_token_idx[i]].tolist())
                
                # Append the best candidates tokens
                full_token_list.append(self.tokens_target_doc[best_candidates[i]][i])

                # Move the text pointer to the next mask
           
                text_pointer = self.target_token_idx[i] + self.len_of_target_masks[i]
                
                if text_pointer >= 4096:
                    break

        # Append the last part of the document        
        full_token_list.append(self.trimmed_target_example['input_ids'][text_pointer:])
        full_token_list = tensor([item for sublist in full_token_list for item in sublist])
        return full_token_list
        
    def custom_collate_fn(self, batch):

        input_ids = [batch['input_ids'].to(dtype = long, device = self.device)]
        attention_masks = [batch['attention_masks'].to(dtype = long, device = self.device)]
        labels = [batch['labels'].to(dtype = long, device = self.device)]

        if True:
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
            labels = pad_sequence(labels, batch_first=True, padding_value=0)

        #assert all(t.shape[0] == input_ids.shape[0] for t in [attention_masks, labels]), \
        #    "Mismatch in tensor sizes after padding."
        return Batch(input_ids = input_ids, attention_masks = attention_masks), labels


    def extraction_attack(self, target_doc_id: int) -> CombinedMetricResult:
        """
        Run the extraction attack on the given documents.
        """
        self.prepare_attack(target_doc_id)
        self.run_attack()
        best_candidate = self.fetch_best_candidates()
        #return self.calculate_metrics()

    def fetch_best_candidates(self) -> list:
        """
        Fetch the best candidates from the confidence scores.
        """
        best_candidates = np.argmax(self.confidence_scores, axis=0)
        return best_candidates

    def trim_example(self, training_ex, labels: list):
        """
        Trim the training example to merge the blanks.
        For example we merge " ... his name is [Masked Firstname] [Masked Lastname] " into "... his name is [Masked]" .
        """

        text_labels = labels
        att_mask = training_ex["attention_masks"]
        indices_to_keep = []
        indices_of_masks = []

        length_of_masks = []
        
        current_mask_id = 0
        tokens_found = 0
        mem = 0
        for i in range(len(text_labels)):

            if text_labels[i] > 0: # If it is a mask

                if current_mask_id == 0: # If it is the first part of a mask
                    indices_to_keep.append(i)
                    indices_of_masks.append(i)
                    current_mask_id = text_labels[i]
                    tokens_found += 1
                    mem = 0

                elif text_labels[i] != current_mask_id + 1: #We've reached a new mask (Since inner parts have id = label+1)
                    length_of_masks.append(mem)
                    indices_to_keep.append(i)
                    indices_of_masks.append(i)
                    current_mask_id = text_labels[i]
                    tokens_found += 1
                    mem = 0

                mem += 1
                
            else:
                # We're outside of a mask, reset the mask id and append the length of the mask
                if current_mask_id != 0:
                    length_of_masks.append(mem)
                    mem = 0
                    current_mask_id = 0

                indices_to_keep.append(i)

            if i>0 and text_labels[i] == -1: 
                # We've reached the padding, stop
                break
       
        # No padding needed here, it's enough to do later on. Could also only return indices to keep.
        modified_example = { 'input_ids' : training_ex["input_ids"][indices_to_keep],  # The ids of the tokens
                            'labels' :text_labels[indices_to_keep],
                            'attention_masks' : att_mask[indices_to_keep],
                            'identifier_types' : [training_ex["identifier_types"][i] for i in indices_to_keep],
                            'offsets' : [training_ex["offsets"][i] for i in indices_to_keep],
                            'entity_types' : [training_ex["entity_types"][i] for i in indices_to_keep] }
        return modified_example, tokens_found, indices_of_masks, length_of_masks

    def fill_blanks(self, text_doc, n_tokens: int, random_sampling = True, action_list = None):
        """
        Fill in the blanks in the text document.
        """

        chosen_tokens = []
        example_len = 4096 # model window size
        initial_token_length = len(text_doc['input_ids'])
        # Initialize the filled example fields with the proper size
        text_ids = zeros(example_len, dtype = long)
        text_labels = zeros(example_len, dtype = long)
        text_offsets = []


        text_entity_types = [[] for i in range(example_len)]
        text_identifier_types = [[] for i in range(example_len)]

        # Fill in the fields with the text document
        text_entity_types[:initial_token_length] = [i for i in text_doc['entity_types']]
        text_identifier_types[:initial_token_length] = [ i for i in text_doc['identifier_types']]
        
        text_att_masks = zeros(example_len, dtype = long)
        
        doc_length = len(text_doc['input_ids'])
        pos = 0
        for i in range(doc_length):

            
            if pos >= example_len:
                break
            if text_doc['labels'][i] > 0: # If we encounter a mask
                b_label = text_doc['labels'][i]
                i_label = b_label+1
                if random_sampling:
                    candidate_text = self.pool.sample_from_candidates(entity_type = text_doc['entity_types'][i])
                else: 
                    candidate_text =self.pool.sample_from_candidates(entity_type = text_doc['entity_types'][i], entity_id = action_list[i])

                tokenized_text = self.tokenizer.encode(candidate_text)[1:-1] # Remove the start and end padding
                chosen_tokens.append(tokenized_text)

                assert pos + len(tokenized_text) <= example_len, \
                    f"Attempting to write {len(tokenized_text)} tokens at pos {pos}, but max size is {example_len}"
                # Ensure no out-of-bounds access
                if pos + len(tokenized_text) >= example_len:
                    break  # Stop filling if we reach the limit

                # Add the tokenized text to the text_ids
                text_ids[pos: pos + len(tokenized_text)] = tensor(tokenized_text)
                text_labels[pos] = b_label
                text_labels[pos+1: pos + len(tokenized_text)] = i_label
                text_att_masks[pos: pos + len(tokenized_text)] = ones(len(tokenized_text), dtype = long)
                text_entity_types[pos: pos + len(tokenized_text)] = [text_doc['entity_types'][i] for j in range(len(tokenized_text))]

                # Dont use these for now
                #text_offsets[pos: pos + len(tokenized_text)] = [text_doc.offsets[i] for j in range(len(tokenized_text))]
                text_identifier_types[pos: pos + len(tokenized_text)] = text_doc['identifier_types'][i]

                pos += len(tokenized_text)

            else:
                text_ids[pos] = text_doc['input_ids'][i]
                text_labels[pos] = text_doc['labels'][i]
                text_att_masks[pos] = text_doc['attention_masks'][i]
                pos += 1


            if text_labels[i] == -1:
                break # should not happen, but just in case

        if pos > example_len:
            logger.warning("The text example is too long, it will be truncated.")
            pos = example_len
        padding_to_add = example_len - pos
        if F.pad(text_ids[:pos], (0,padding_to_add), value = self.tokenizer.pad_token_id).shape[0] != example_len:
            logger.warning("The padding is not done correctly, the tensor text_pos is not of the correct shape.")
        
        if F.pad(text_labels[:pos], (0,padding_to_add), value = -1).shape[0] != example_len:
            logger.warning("The padding is not done correctly, the tensor text_labels is not of the correct shape.")

        if F.pad(text_att_masks[:pos], (0,padding_to_add), value = -1).shape[0] != example_len:
            logger.warning("The padding is not done correctly, the tensor text_att_masks is not of the correct shape.")


        # Prolly need to make sure that the padding is done correctly, tensors and on device
        filled_text_example ={ 'input_ids' : F.pad(text_ids[:pos], (0,padding_to_add), value = self.tokenizer.pad_token_id),
                            'labels' : F.pad(text_labels[:pos], (0,padding_to_add), value = -1),
                            'attention_masks' : F.pad(text_att_masks[:pos], (0,padding_to_add), value = -1),
                            'identifier_types' : text_identifier_types[:pos] + padding_to_add*[" "],
                            'offsets' : text_offsets[:pos] + padding_to_add*[-1],
                            'entity_types' : text_entity_types[:pos] + padding_to_add*["O"] }
        

        return filled_text_example, chosen_tokens
        
    

    # Abstract methods
    def description(self) -> dict:
        """Return a description of the attack."""
        title_str = "Mastermind Attack"
        reference_str = " - "
        summary_str = "Mastermind attack works like a black pin mastermind game, where the pin is a masked PII with known class."
        detailed_str = "The attack is executed according to: \
            1. Randomly sample replacement strings to fill the blanks in a masked document.\
            2. Calculate the confidence scores for the replacements.\
            3. Repeat the process for a number of times.\
            4. Choose the best candidate based on the confidence scores.\
            "
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }
