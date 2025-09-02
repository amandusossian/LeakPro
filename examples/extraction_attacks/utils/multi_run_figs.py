import matplotlib.pyplot as plt
import os
import sys
import numpy as np



def gen_entity_conversion_dicts():
    entity_string_to_int  = {
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
    entity_int_to_string = {v: k for k, v in entity_string_to_int.items()}
    return entity_int_to_string, entity_string_to_int


def generate_avg_plots(attack_obj):
    """
    Here we generate the average confidence of a specific mask type

    We first extract the sum of all confidences (avg per doc) for each fraction of 
    other pii being correct.

    Then these are plotted. 


    """
    path_to_datafolder = path
    files_to_load = [f for f in os.listdir(path_to_datafolder)]
    
    
    
    entity_int_to_string, entity_string_to_int = gen_entity_conversion_dicts()
    conf_target = {}
    times_seen_fraction = {}

   
    target_n_masks_list = []
    remaining_n_masks_list = []
    n_runs = len(files_to_load)
    
    # For each file (i.e. run on document)
    for k, file in enumerate(files_to_load):
        loaded_res = np.load( path + '/' + file, allow_pickle = True )

        current_res_dict = loaded_res.item()
        confidences = current_res_dict['observed_rewards']
       
        n_masks_of_entity_type = current_res_dict['n_masks_of_target_type']
        n_masks_in_target = current_res_dict['n_masks_in_target']
        n_remaining_masks = n_masks_in_target - n_masks_of_entity_type
        entity_types_of_all_masks = current_res_dict['entity_types']
       
        target_entity_type = current_res_dict['target_entity_type']
   
        fractions = current_res_dict['fractions']
       
        doc_id = current_res_dict['doc_id']

        

        print(f'Processing file nr {k+1}, ran on doc id: {doc_id}')
        # For all the fractions of remaining masks set to correct
        for j in range(len(fractions)):
            
            # This is the current fraction of the remaining (non-target) masks which were set correct 
            fraction = fractions[j]

            # Current confidences of all of the masks
            current_confidences = confidences[j]

            # init/reset tmp values
            tmp_conf_others = 0
            tmp_conf_target = 0

            # For all of the tokens in the document:
            for iMask, confMask in enumerate(current_confidences):
                
                # If we're at a token belonging to the target entity type
                if entity_types_of_all_masks[iMask] == target_entity_type:
                    tmp_conf_target += confMask # Add the confidence of the target mask
                
                # Or one of the others
                else: 
                    tmp_conf_others += confMask


            # These are not used for now, could be later on. 
            target_n_masks_list.append(n_masks_of_entity_type)
            remaining_n_masks_list.append(n_remaining_masks)
            
            # If we've seen the current fraction already, we need to add to it and later on divide by the number of times it was investigated!
            if fraction in conf_target.keys():
                conf_target[fraction] += tmp_conf_target / n_masks_of_entity_type # Adds the average confidence of the target masks for the specific run
                times_seen_fraction[fraction] += 1 # Add 1 to the number of times we've seen the current fraction investigated

            # Otherwise, initiate it
            else: 
                conf_target[fraction] = tmp_conf_target / n_masks_of_entity_type
                times_seen_fraction[fraction] = 1
            
            
    
    #Used for plotting
    plot_fracs = []
    plot_confs = []
    
    # conf_target is a dict with 
    # Keys: fraction of non-target type masks are set correct
    # Vals: List of confidences for that fraction value per run over document

    # Extract the fractions to a list and divide the mean of the confidences with the number of times we've seen that fraction.
    for total_fraction, total_confidences in conf_target.items():
        plot_fracs.append(total_fraction)
        plot_confs.append(np.mean(total_confidences) / times_seen_fraction[total_fraction])
        
        
    plt.plot(plot_fracs, plot_confs)
    plt.grid()
    entity_string = entity_int_to_string[target_entity_type]
    plt.title(f'Average confidence over {n_runs} runs for entity type {entity_string}') 
    print(entity_string)
    plt.xlabel('Fraction of remaining PII forced correct')
    plt.ylabel('Confidence')
    plt.savefig(f'./multi_run_avg_entity_{str(entity_string)}.png')
    


generate_avg_plots('./outputs/tests/multi_run')