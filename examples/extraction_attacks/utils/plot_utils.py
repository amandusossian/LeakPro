import matplotlib.pyplot as plt
import numpy as np
import os

def plot_info_per_pool(attack_obj):
    for iMask, idxMask in enumerate(attack_obj.indices_of_target_masks):
       
        curr_entity_type = attack_obj.true_entity_types[iMask].item()
        corr_id = attack_obj.pool.true_idxs[iMask]
        corr_id_conf = []
        incorr_id_conf = []
        dct = {}
        max_val = -1
        max_val_idx = -1
        
        for iAttempt in range(attack_obj.n_evaluations):

            selected_action = attack_obj.actions_taken[iAttempt][iMask]
            curr_conf = attack_obj.confidence_scores[iAttempt][iMask]
            
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


            
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.set_title(f"Confidence scores for PII #{iMask}, Entity type {attack_obj.entity_int_to_string[attack_obj.true_entity_types[iMask].item()]}")
        ax.hist(corr_id_conf, bins=50, color='blue', alpha=0.5, label="Correct action")
        ax.hist(incorr_id_conf, bins=50, color='red', alpha=0.5, label="Incorrect action")
        for c in corr_id_conf:
            ax.axvline(x=c, color='blue', alpha=0.5, linestyle='--')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig.tight_layout()
        fig.savefig( attack_obj.figpath + f"/Conf_hist_Mask_{iMask}.svg", bbox_inches='tight')
        plt.close(fig)


        text_list = []
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.tight_layout()

        ax.set_title(f"Confidence scores for PII #{iMask} per action, entity type {attack_obj.entity_int_to_string[attack_obj.true_entity_types[iMask].item()]}, true entity: {attack_obj.tokenizer.decode(attack_obj.true_input_ids[iMask])}")
        for iAction in sorted(dct.keys()):
            n_tries = len(dct[iAction])
            # Sizes increase as time goes on:
            sizes = np.linspace(10, 50, n_tries) 
            if iAction == 0:
                ax.scatter(iAction*np.ones(n_tries), dct[iAction], s = sizes, label=f"Correct action ({n_tries} tries)", color='blue', zorder = 2)
            else: 
                ax.scatter(iAction*np.ones(n_tries), dct[iAction], s = sizes,  label=f"Action {iAction} ({n_tries} tries)", zorder = 2)
            text_list.append(attack_obj.tokenizer.decode(attack_obj.pool.local_pool[iMask][iAction]))
        ax.scatter(max_val_idx*np.ones(1), max_val, s = 125, label="Best action", marker="D", color='yellow', edgecolors='black', zorder = 3)
        show_true_conf = False
        if show_true_conf:
            ax.scatter(corr_id*np.ones(1), attack_obj.true_conf[iMask], s = 125, label="True, in fully corr doc", marker="D", color='green', edgecolors='black', zorder = 3)
        ax.grid(zorder = -1)

        ax.set_xlabel("Action taken")
        ax.set_ylabel("Confidence score")
        
        ax.set_xticks(range(len(text_list)))

        ax.set_xticklabels(text_list, rotation=70)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        fig.savefig(attack_obj.figpath + f"/Conf_per_action_Mask_{iMask}_doc_{attack_obj.target_doc_idx}.svg", bbox_inches='tight')
        plt.close()

        if iMask>=7: 
            break 



def plot_histograms(attack_obj):

    
    epsilon = 0.001
    
    colors_single = {
            "corr_pred": '#3cb44b',  # vivid green
            "incorr_pred": '#e6194b',  # vibrant red-pink
            "true_confidences": '#4363d8' ,
            
        }

    best_candidates = np.argmax(attack_obj.observed_rewards, axis=0)

    correct_confidences = []
    incorrect_confidences = []
    all_confidences = []
    validation_confidences = attack_obj.true_conf
    true_confidences = [i**attack_obj.tsallis_reward_factor for i in validation_confidences] if attack_obj.attack_strategy == "bandit" else validation_confidences
    
    for i in range(attack_obj.n_masks_in_target):
        if attack_obj.binary_correct_guesses[i]:
            correct_confidences.append(attack_obj.observed_rewards[best_candidates[i]][i] + epsilon)
        else:
            incorrect_confidences.append(attack_obj.observed_rewards[best_candidates[i]][i] + epsilon)
        all_confidences.append(attack_obj.observed_rewards[best_candidates[i]][i] + epsilon)

    # Make uniform bins

    
    n_bins = 100
    min_bin = min(incorrect_confidences + correct_confidences + validation_confidences)
    max_bin = max(incorrect_confidences + correct_confidences + validation_confidences)
    

    # Combined histogram
    bins_ = np.linspace(min_bin-0.001, max_bin+0.01, n_bins)
    
    
    gen_pool_dist_fig = True
    if gen_pool_dist_fig: 
        plt.figure(figsize=(10, 5))
        pool_dist = {}
        for i in attack_obj.true_entity_types:
            cet = i.item()
            if cet not in pool_dist:
                pool_dist[cet] = 0
            pool_dist[cet] += 1
        plt.vlines(pool_dist.keys(), pool_dist.values(), 0, linewidth = 6, color='blue', alpha=0.75)
        plt.xlabel("Entity type")
        plt.ylabel("Number of entities in pool")
        plt.title("Distribution of masked entity types in the document")
        plt.grid()
        plt.xticks(list(pool_dist.keys()), [attack_obj.entity_int_to_string[i] for i in pool_dist.keys()])
        plt.savefig(attack_obj.figpath + f"/pool_distribution_doc_{attack_obj.target_doc_idx}.svg")
        plt.close()

    generate_single_hist = True
    if generate_single_hist:
        plt.figure(figsize=(10, 5))
        
        doitstacked = False
        
        plt.hist([incorrect_confidences, true_confidences, correct_confidences], 
                bins = bins_,  
                color=[colors_single["incorr_pred"], colors_single["true_confidences"],colors_single["corr_pred"]], 
                stacked=doitstacked, 
                alpha=0.5, 
                label=["Identified incorrect", "True confidences", "Identified correct"])    
    


        plt.xlabel("Confidence Score")
        plt.xlim(-0.01, 1.01)
        plt.ylabel("Frequency")
        title_string =f"Confidence Scores Distribution, {attack_obj.final_n_correct} correct ({attack_obj.final_n_almost_correct} almost correct) out of {attack_obj.n_masks_in_target}, using {attack_obj.attack_strategy} sampling"
        if attack_obj.attack_strategy == "bandit":
            title_string += f" reward factor {attack_obj.bandit_parameters['tsallis_reward_factor']}"
        plt.title(title_string)
        plt.legend()
        plt.grid()      
        plt.savefig(attack_obj.figpath + "/confidence_scores_combined.svg")
        plt.close()




    plot_entity_wise_histograms = True
    if plot_entity_wise_histograms:
        # Individual histograms

        # Generate colors for the histograms, per entity type
        data_dists = {}
        true_dists = {}
        entity_colors = {
            0: '#1f77b4',  # blue
            1: '#ff7f0e',  # orange
            2: '#2ca02c',  # green
            3: '#d62728',  # red
            4: '#9467bd',  # purple
            5: '#8c564b',  # brown
            6: '#e377c2',  # pink
            7: "#616161"   # gray
        }


        for i in range(attack_obj.n_masks_in_target):
            # Fetch the entity type
            entity_type = attack_obj.true_entity_types[i].item()

            # Append the confidence scores to the corresponding entity type, for the 
            # best estimate and true conf
            if entity_type not in data_dists:
                data_dists[entity_type] = []
                true_dists[entity_type] = []


            data_dists[entity_type].append(attack_obj.observed_rewards[best_candidates[i]][i]+epsilon)
            true_dists[entity_type].append(attack_obj.true_conf[i]+epsilon)

        combined = []
        for key in data_dists:
            combined.extend(data_dists[key])
            combined.extend(true_dists[key])
        combined = np.array(combined)
        bins_2 = np.histogram_bin_edges(combined, bins=50, range=(-0.01, 1.01))

        figure, axes = plt.subplots(2, 1, figsize=(10, 5))
        
        axes[0].set_title("Confidence score distribution per entity type of best guess")
        axes[0].set_xlabel("Confidence Score")
        axes[0].set_ylabel("Count")
        #axes[0].set_xlim(-0.01, 1.01)
        
        axes[0].grid()

        axes[1].set_title("Confidence score distribution per entity type of true document")
        axes[1].set_xlabel("Confidence Score")
        axes[1].set_ylabel("Count")
        #axes[1].set_xlim(-0.01, 1.01)
        
        axes[1].grid()
        l1 = 0
        l2 = 0
        key_list = sorted(list(data_dists.keys()))
        
            
        
        axes[0].hist([data_dists[key] for key in key_list], 
                        bins= bins_2, 
                        color = [entity_colors[key] for key in key_list], 
                        alpha=0.6, 
                        stacked = True, 
                        label=[f"{attack_obj.entity_int_to_string[key]}" for key in key_list], 
                        density=False)    
        
        axes[1].hist([true_dists[key] for key in key_list], 
                        bins= bins_2,
                        color = [entity_colors[key] for key in key_list], 
                        alpha=0.6, 
                        label=[f"{attack_obj.entity_int_to_string[key]}" for key in key_list],
                        stacked = True,
                        density=False)

        axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.show()
        plt.savefig(attack_obj.figpath + f"/confidence_scores_comparison_estimate_and_truth_doc_{attack_obj.target_doc_idx}.svg")
        plt.close()



def plot_single_mask_results(attack_obj):
    """
    Plots confidence over time as a function of how many guesses are correct
    """
    confidence = [attack_obj.confidence_scores[i][attack_obj.SMI_mask_idx] for i in range(attack_obj.n_evaluations)] # Extracts the confidence of the specific sample
    n_correct_masks = [i for i in attack_obj.n_corr_per_guess]

    sorted_indices = np.argsort(n_correct_masks)
    x_plot = [n_correct_masks[i] for i in sorted_indices]
    y_plot = [confidence[i] for i in sorted_indices]
    plt.plot(x_plot, y_plot, 'o')
    plt.title(f'Confidence of mask nr {attack_obj.SMI_mask_idx} as number of correct masks increases. \nIn full document, {attack_obj.true_conf[attack_obj.SMI_mask_idx]:.6f} ')
    plt.xlabel('number of forced correct masks')
    plt.ylabel('Confidence')
    plt.grid()
    plt.show()
    plt.savefig(attack_obj.figpath + '/single_conf.svg')

def plot_expanding_window_result(attack_obj):
    
    confidences = np.zeros(attack_obj.n_evaluations)
    


    # Average across the PII in case of avg over all. Otherwise just extract it. 
    if attack_obj.EW_centered_single: 
        plot_EW_centered_results(attack_obj)
        return
    
    for i in range(attack_obj.n_evaluations):

        tmp_conf = 0.0
        n_entities = 0

        for j in range(len(attack_obj.confidence_scores[i])):

            if attack_obj.EW_single_entity:
            
                if j == attack_obj.EW_single_entity_actual_mask_idx:    
                    tmp_conf = attack_obj.confidence_scores[i][j]
                    break 

            else:
                if attack_obj.true_entity_types[j] == attack_obj.EW_entity_type:
                    tmp_conf += attack_obj.confidence_scores[i][j]
                    n_entities += 1
            
        if attack_obj.EW_single_entity: 
            confidences[i] = tmp_conf
        
        else: 
            if n_entities>1: # Avg over all entities of the PII class if applicable
                confidences[i] = tmp_conf / n_entities        
            else: 
                confidences[i] = tmp_conf 


    window_size = attack_obj.EW_window_expansions

    # Average over window sizes, i.e. group all confidences of the same window size and get the avg. Just to create a nice plot.
    window_plot = []
    conf_plot = []
    mem_window_size = attack_obj.EW_original_size
    mem_seen_per_window = 0
    mem_conf = 0
    for i, w in enumerate(window_size):
        if w != mem_window_size:
            conf_plot.append(mem_conf/mem_seen_per_window)
            window_plot.append(mem_window_size)
            mem_window_size = w
            mem_conf = confidences[i]
            mem_seen_per_window = 1
        else: 
            mem_seen_per_window += 1
            mem_conf += confidences[i]
            
    conf_plot.append(mem_conf/mem_seen_per_window)
    window_plot.append(mem_window_size)
    

    plt.plot(window_plot, conf_plot, label = 'Mean')
    plt.plot(window_size, confidences, 'o', alpha = 0.2, label = f'Score per eval' if attack_obj.EW_single_entity else f'Class avg score per eval')
    plt.title(f'Confidence as window size increases. Entity type {attack_obj.entity_int_to_string[attack_obj.EW_entity_type]}, \n{f"Confidence of a single PII,  [ {attack_obj.tokenizer.decode(attack_obj.true_input_ids[attack_obj.EW_single_entity_actual_mask_idx]) } ]." if attack_obj.EW_single_entity else "Avg over all PII in the class."} ')
    plt.xlabel('Window size')
    plt.ylabel('Confidence')
    #plt.ylim(0, 1.1)
    #plt.yticks(np.linspace(0,1,11))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(attack_obj.figpath + f"/EW_{attack_obj.entity_int_to_string[attack_obj.EW_entity_type]}_{'single' if attack_obj.EW_single_entity else 'all'}_PII{'rest_random' if not attack_obj.EW_set_rest_correct else ''}.svg")



def plot_EW_centered_results(attack_obj):

    
    original_confidence_of_mask = attack_obj.true_conf[attack_obj.EW_single_entity_actual_mask_idx]
    plt.plot( attack_obj.EW_window_expansions, attack_obj.observed_rewards, label = 'Confidence')
    plt.hlines(original_confidence_of_mask,xmin = attack_obj.EW_window_expansions[0], xmax = attack_obj.EW_window_expansions[-1], color = 'orange', linestyles='dashed', label = 'Conf of true mask in full doc' )


    true_seq = attack_obj.true_input_ids[attack_obj.EW_single_entity_actual_mask_idx]
    true_string = attack_obj.tokenizer.decode(true_seq)
    
    if attack_obj.EW_single_entity_replace_target_mask_bool:
        mask_seq = attack_obj.EW_replacement_sequence
        mask_string = attack_obj.tokenizer.decode(mask_seq)
        
        rep_title_string = f'Mask "{true_string}" replaced by "{mask_string}"\n'  if true_string != mask_string else f'Mask "{mask_string}"\n'
   
    non_rep_title_string = f'Mask "{true_string}"\n' 
    entity_string = attack_obj.entity_int_to_string[attack_obj.EW_entity_type_int]

    plt.title( (
        f'Confidence as window size increases. Entity type {entity_string} \n'
        f' {rep_title_string if attack_obj.EW_single_entity_replace_target_mask_bool else non_rep_title_string}'
        f' Mask idx nr {attack_obj.EW_single_entity_actual_mask_idx} in document.'
    ))
    
    
    plt.xlabel('Size of Window')
    plt.ylabel('Confidence')
    #plt.text(0, original_confidence_of_mask+0.005, 'Original confidence', color = 'black')
    # How to make a good showing of when the full document is reached? Should be at the end regardless, but there's a difference between reaching all the way back to the start and to the end
    #plt.vlines(attack_obj.doc_token_length, np.min(attack_obj.observed_rewards), np.max(attack_obj.observed_rewards), color = 'black', linestyles='dashed', alpha = 0.5, label = 'Padding Reached')
    plt.grid()
    plt.legend(loc = 'lower right')
    plt.tight_layout()

    plt.savefig(attack_obj.figpath + f"/EW_centered_{attack_obj.entity_int_to_string[attack_obj.EW_entity_type_int]}_doc_{attack_obj.target_doc_idx}_mask_pos_{attack_obj.EW_single_entity_actual_mask_idx}{f'_replaced' if attack_obj.EW_single_entity_replace_target_mask_bool else '_org'}.svg")


  

def plot_MR_avg_results(attack_obj):
    """
    Here we generate the average confidence of a specific mask type

    We first extract the sum of all confidences (avg per doc) for each fraction of 
    other pii being correct.

    Then these are plotted. 


    """
    path_to_datafolder = attack_obj.MR_path
    files_to_load = [f for f in os.listdir(path_to_datafolder)]
    
    
    
    entity_int_to_string = attack_obj.entity_int_to_string
    conf_target = {}
    times_seen_fraction = {}
    target_entity_type = 0
   
    target_n_masks_list = []
    remaining_n_masks_list = []
    n_runs = len(files_to_load)
    
    # For each file (i.e. run on document)
    for k, file in enumerate(files_to_load):
        loaded_res = np.load( path_to_datafolder + '/' + file, allow_pickle = True )

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
    plt.title(f'Average confidence over {n_runs} run(s) for entity type {entity_string}') 
    print(entity_string)
    plt.xlabel('Fraction of remaining PII forced correct')
    plt.ylabel('Confidence')
    plt.savefig( attack_obj.MR_path+ f'/multi_run_avg_entity_{str(entity_string)}.png')
    

def plot_gather_stats_results(attack_obj, two_mask_experiment, n_seen_masks, n_diff_docs, entity_wise_probs, true_probs_of_all):
    """
    Plots the histograms of the true probabilities of the entities in the pool
    """
    
    if two_mask_experiment: 
        
        true_probs_of_all = [prob for doc in true_probs_of_all for prob in doc] # Flattening of a nested list. 
        
        # Make histogram of the true confidences
        plt.figure()
        plt.hist(true_probs_of_all, bins = 100, density = True, alpha = 0.5, label = 'Confidences')
        title_string1 = f'True Confidences of the masks in the full documents. \n{n_seen_masks} masks, taken from {n_diff_docs} documents.'
        plt.title(title_string1)
        plt.grid()     
        plt.legend()
        plt.savefig( attack_obj.figpath + '/true_probs_histogram.svg')
        
        # Make histogram of confidences of context free masks
        entity_labels, entity_hist = zip(*entity_wise_probs.items())
        entity_labels = [attack_obj.entity_int_to_string[el] for el in entity_labels]


        plt.figure()
        plt.hist([e for entity_conf in entity_hist for e in entity_conf], bins = 100, density = True, alpha = 0.5, label = 'Confidences')
        plt.grid()        
        plt.legend()
        title_string2 = f'Context-free confidences of the masks. \n {n_seen_masks} masks, taken from {n_diff_docs} documents.'
        plt.title(title_string2)
        plt.savefig( attack_obj.figpath + '/context_free_confidence_histogram.svg')

        
        plt.figure()
        plt.hist(entity_hist, bins = 100, density = True, stacked = True, alpha=0.5, label = entity_labels)
        title_string3 = f'Context-free confidences of the masks by entity type. \n {n_seen_masks} masks, taken from {n_diff_docs} documents.'
        plt.title(title_string3)
        plt.legend()
        plt.grid() 
        plt.savefig( attack_obj.figpath + '/context_free_confidence_entity_wise_histogram.svg')

        filtered_pairs = [( hist, label) for label, hist in zip(entity_labels, entity_hist) if label != "DATETIME"]
        entity_hist_no_datetime, entity_labels_no_datetime = zip(*filtered_pairs)

        plt.figure()
        plt.hist(entity_hist_no_datetime, bins = 100, density = True, stacked = True, alpha=0.5, label = entity_labels_no_datetime)
        title_string4 = f'Context-free confidences of the masks by entity type, without DATETIME. \n {n_seen_masks} masks, taken from {n_diff_docs} documents.'
        plt.title(title_string4)
        plt.legend()
        plt.grid() 
        plt.savefig( attack_obj.figpath + '/context_free_confidence_entity_wise_histogram_no_datetime.svg')
