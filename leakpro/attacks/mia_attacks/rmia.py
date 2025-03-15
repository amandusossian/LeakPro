"""Implementation of the RMIA attack."""

import numpy as np

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger
from collections import Counter


class AttackRMIA(AbstractMIA):
    """Implementation of the RMIA attack."""

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the RMIA attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(handler)
        self.shadow_models = []
        self.signal = ModelLogits()
        self.epsilon = 1e-6
        self.shadow_models = None
        self.shadow_model_indices = None

        logger.info("Configuring RMIA attack")
        self._configure_attack(configs)


    def _configure_attack(self:Self, configs: dict) -> None:
        """Configure the RMIA attack.

        Args:
        ----
            configs (dict): Configuration parameters for the attack.

        """
        self.num_shadow_models = configs.get("num_shadow_models", 4)
        self.offline_a = configs.get("data_fraction", 0.33)
        self.offline_b = configs.get("offline_b", 0.66)
        self.gamma = configs.get("gamma", 2.0)
        self.temperature = configs.get("temperature", 2.0)
        self.training_data_fraction = configs.get("training_data_fraction", 0.5)
        self.online = configs.get("online", False)
        self.attack_data_fraction = configs.get("attack_data_fraction", 0.1)
        self.avg_method = configs.get("avg_method", 'avg_over_masked')

        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            "num_shadow_models": (self.num_shadow_models, 1, None),
            "offline_a": (self.offline_a, 0, 1),
            "offline_b": (self.offline_b, 0, 1),
            "gamma": (self.gamma, 0, None),
            "temperature": (self.temperature, 0, None),
            "training_data_fraction": (self.training_data_fraction, 0, 1),
            "attack_data_fraction": (self.attack_data_fraction, 0, 1),
        }

        # Validate parameters
        for param_name, (param_value, min_val, max_val) in validation_dict.items():
            self._validate_config(param_name, param_value, min_val, max_val)

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "RMIA attack"
        reference_str = "Zarifzadeh, Sajjad, Philippe Cheng-Jie Marc Liu, and Reza Shokri. \
            Low-Cost High-Power Membership Inference by Boosting Relativity. (2023)."
        summary_str = "The RMIA attack is a membership inference attack based on the output logits of a black-box model."
        detailed_str = "The attack is executed according to: \
            1. A fraction of the population is sampled to compute the likelihood LR_z of p(z|theta) to p(z) for the target model.\
            2. The ratio is used to compute the likelihood ratio LR_x of p(x|theta) to p(x) for the target model. \
            3. The ratio LL_x/LL_z is viewed as a random variable (z is random) and used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """
        logger.info("Preparing shadow models for RMIA attack")
        # Check number of shadow models that are available

        # sample dataset to compute histogram
        logger.info("Preparing attack data for training the RMIA attack")

        # Get all available indices for attack dataset, if self.online = True, include training and test data
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
                                                                       include_test_indices = self.online)

        # train shadow models
        logger.info(f"Check for {self.num_shadow_models} shadow models (dataset: {len(self.attack_data_indices)} points)")
        
    
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models = self.num_shadow_models,
            shadow_population = self.attack_data_indices,
            training_fraction = self.training_data_fraction,
            online = self.online)
        # load shadow models
        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        # compute quantities that are not touching the audit dataset
        if self.online is False:
            # compute the ratio of p(z|theta) (target model) to p(z)=sum_{theta'} p(z|theta') (shadow models)
            # for all points in the attack dataset output from signal: # models x # data points x # classes

            # subsample the attack data based on the fraction
            logger.info(f"Subsampling attack data from {len(self.attack_data_indices)} points")
            n_points = int(self.attack_data_fraction * len(self.attack_data_indices))
            n_points = min(n_points, len(self.handler.test_indices))
            chosen_attack_data_indices = np.random.choice(self.handler.test_indices, n_points, replace=False)

            #chosen_attack_data_indices = np.random.choice(self.attack_data_indices, n_points, replace=False)
            logger.info(f"Number of attack data points after subsampling: {len(chosen_attack_data_indices)}")

            # get the true label indices
            z_true_labels = self.handler.get_labels(chosen_attack_data_indices)
            assert np.issubdtype(z_true_labels.dtype, np.integer)

            # run points through real model to collect the logits
            logits_theta = np.array(self.signal([self.target_model], self.handler, chosen_attack_data_indices))
            # collect the softmax output of the correct class
            n_attack_points = len(chosen_attack_data_indices)
            p_z_given_theta = softmax_logits(logits_theta, self.temperature)[np.arange(n_attack_points)[:,None], 
                                                                                np.arange(z_true_labels.shape[1]), 
                                                                                z_true_labels]

            # Converts from token level to doc level
            p_z_given_theta = self.token_to_doc_lvl(p_z_given_theta, z_true_labels)

            # run points through shadow models and collect the logits
            logits_shadow_models = self.signal(self.shadow_models, self.handler, chosen_attack_data_indices)
            # collect the softmax output of the correct class for each shadow model
            sm_logits_shadow_models = [softmax_logits(x, self.temperature) for x in logits_shadow_models]

            p_z_given_shadow_models = np.array([x[np.arange(n_attack_points)[:,None], 
                                              np.arange(z_true_labels.shape[1]), 
                                              z_true_labels] for x in sm_logits_shadow_models])
        
            # Convert from token level to doc level
            p_z_given_shadow_models = np.array([self.token_to_doc_lvl(x, z_true_labels) for x in p_z_given_shadow_models])

            # evaluate the marginal p(z)
            p_z = np.mean(p_z_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_z_given_shadow_models.squeeze()
            p_z = 0.5*((self.offline_a + 1) * p_z + (1-self.offline_a))

            self.ratio_z = p_z_given_theta / (p_z + self.epsilon)

    def _online_attack(self:Self) -> None:
        logger.info("Running RMIA online attack")

        # STEP 1: find out which audit data points can actually be audited
        # find the shadow models that are trained on what points in the audit dataset
        in_indices_mask = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"]).T
        # filter out the points that no shadow model has seen and points that all shadow models have seen
        num_shadow_models_seen_points = np.sum(in_indices_mask, axis=0)
        # make sure that the audit points are included in the shadow model training (but not all)
        mask = (num_shadow_models_seen_points > 0) & (num_shadow_models_seen_points < self.num_shadow_models)

        # STEP 2: Select datapoints that are auditable
        audit_data_indices = self.audit_dataset["data"][mask]
        self.audit_data_indices = audit_data_indices

        self.n_in_members = np.sum(mask[self.audit_dataset["in_members"]])
        # find out how many in-members survived the filtering
        in_members = np.arange(self.n_in_members)
        # find out how many out-members survived the filtering
        self.n_out_members = np.sum(mask[self.audit_dataset["out_members"]])
        
        out_members = np.arange(self.n_in_members, self.n_in_members + self.n_out_members)

        # get the true labels for the audit indices
        ground_truth_indices = self.handler.get_labels(audit_data_indices).astype(int)

        assert len(audit_data_indices) == len(ground_truth_indices)
        assert len(audit_data_indices) == len(in_members) + len(out_members)

        if len(audit_data_indices) == 0:
            raise ValueError("No points in the audit dataset are used for the shadow models")

        out_model_indices = ~in_indices_mask[:,mask]

        logger.info(f"Number of points in the audit dataset that are used for online attack: {len(audit_data_indices)}")

        # STEP 3: Run the attack
        # run points through target model to get logits
        logits_theta = np.array(self.signal([self.target_model], self.handler, audit_data_indices)).squeeze()
        # collect the softmax output of the correct class
        n_audit_points = len(audit_data_indices)
        
        # Original: 
        # p_x_given_target_model = softmax_logits(logits_theta, self.temperature)[np.arange(n_audit_points),:,ground_truth_indices]

        p_x_given_target_model = softmax_logits(logits_theta, self.temperature)[np.arange(n_audit_points)[:,None], 
                                                                                np.arange(ground_truth_indices.shape[1]), 
                                                                                ground_truth_indices]

        # Converts the token-level logits to document level instead
        p_x_given_target_model = self.token_to_doc_lvl(p_x_given_target_model, ground_truth_indices)
        

        # run points through shadow models, colelct logits and compute p(x)
        logits_shadow_models = self.signal(self.shadow_models, self.handler, audit_data_indices)
        sm_shadow_models = [softmax_logits(x, self.temperature) for x in logits_shadow_models]

        p_x_given_shadow_models = np.array([x[np.arange(n_audit_points)[:,None],
                                              np.arange(ground_truth_indices.shape[1]), 
                                              ground_truth_indices] for x in sm_shadow_models])
        
        # Converts the token-level logits to document level instead
        p_x_given_shadow_models = [self.token_to_doc_lvl(x, ground_truth_indices) for x in p_x_given_shadow_models]
        

        p_x = np.mean(p_x_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_x_given_shadow_models.squeeze()
        # compute the ratio of p(x|theta) to p(x)
        ratio_x = p_x_given_target_model / (p_x + self.epsilon)

        # Make a "random sample" to compute p(z) for points in attack dataset on the OUT shadow models for each audit point
        self.attack_data_index = self.sample_indices_from_population(include_train_indices = False,
                                                                     include_test_indices = False)
        if len(self.attack_data_index) == 0:
            raise ValueError("There are no auxilliary points to use for the attack.")

        # subsample the attack data based on the fraction
        logger.info(f"Subsampling attack data from {len(self.attack_data_index)} points")
        self.attack_data_index = np.random.choice(
            self.attack_data_index,
            int(self.attack_data_fraction * len(self.attack_data_index)),
            replace=False
        )
        logger.info(f"Number of attack data points after subsampling: {len(self.attack_data_index)}")

        # get the true label indices
        z_true_labels = self.handler.get_labels(self.attack_data_index).astype(int)
        assert np.issubdtype(z_true_labels.dtype, np.integer)

        # run points through real model to collect the logits
        logits_target_model = np.array(self.signal([self.target_model], self.handler, self.attack_data_index))
        # collect the softmax output of the correct class
        n_attack_points = len(self.attack_data_index)
        p_z_given_target_model = softmax_logits(logits_target_model, self.temperature)[np.arange(n_attack_points)[:,None], 
                                                                                np.arange(z_true_labels.shape[1]), 
                                                                                z_true_labels]

        # Converts the token-level logits to document level instead
        p_z_given_target_model = self.token_to_doc_lvl(p_z_given_target_model, z_true_labels)

        # run points through shadow models and collect the logits
        logits_shadow_models = self.signal(self.shadow_models, self.handler, self.attack_data_index)
        # collect the softmax output of the correct class for each shadow model
        sm_shadow_models = [softmax_logits(x, self.temperature) for x in logits_shadow_models]
        p_z_given_shadow_models = np.array([x[np.arange(n_attack_points)[:,None], 
                                              np.arange(z_true_labels.shape[1]), 
                                              z_true_labels] for x in sm_shadow_models])
        
        p_z_given_shadow_models = np.array([self.token_to_doc_lvl(x, z_true_labels) for x in p_z_given_shadow_models])
        
        # evaluate the marginal p(z) by averaging over the OUT models
        p_z = np.zeros((len(audit_data_indices), len(self.attack_data_index)))
        for i in range(len(audit_data_indices)):
            model_mask = out_model_indices[:,i]
            p_z[i] = np.mean(p_z_given_shadow_models[model_mask, :], axis = 0)
        
        ratio_z = p_z_given_target_model/ (p_z + self.epsilon)

        # for each x, compute the score
        likelihoods = ratio_x[...,np.newaxis] / ratio_z
        score = np.mean(likelihoods > self.gamma, axis=1)

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)


    def token_to_doc_lvl(self, token_lvl_logits, token_lvl_labels):
        """
        Converts token level logits to a document level insted. 

        avg_over_masked only takes the mean over the lables which are not set to 'unmasked'.
        avg_over_all takes mean across all predictions in the document.
        """
        doc_logits = np.zeros(shape = token_lvl_labels.shape[:-1])
        avg_method = self.avg_method
        
        if avg_method == 'avg_over_masked':
            for i in range(token_lvl_labels.shape[0]):
                mask = token_lvl_labels[i] > 0
                doc_logits[i] = np.mean(token_lvl_logits[i][mask])
        elif avg_method == 'avg_over_all':
            doc_logits = np.mean(token_lvl_logits, axis = 1)

        return doc_logits


    def _offline_attack(self:Self) -> None:
        logger.info("Running RMIA offline attack")

        # run target points through real model to get logits
        logits_theta = np.array(self.signal([self.target_model], self.handler, self.audit_dataset["data"]))
        # collect the softmax output of the correct class
        ground_truth_indices = self.handler.get_labels(self.audit_dataset["data"])
        assert np.issubdtype(ground_truth_indices.dtype, np.integer)

        n_audit_points = len(self.audit_dataset["data"])
        p_x_given_target_model = softmax_logits(logits_theta, self.temperature)[np.arange(n_audit_points)[:,None], 
                                                                                np.arange(ground_truth_indices.shape[1]), 
                                                                                ground_truth_indices]

        # Converts the token-level logits to document level instead
        p_x_given_target_model = self.token_to_doc_lvl(p_x_given_target_model, ground_truth_indices)

        # run points through shadow models and collect the logits
        logits_shadow_models = self.signal(self.shadow_models, self.handler, self.audit_dataset["data"])
        # collect the softmax output of the correct class for each shadow model
        # Stack to dimension # models x # data points
        sm_shadow_models = [softmax_logits(x, self.temperature) for x in logits_shadow_models]
        p_x_given_shadow_models = np.array([x[np.arange(n_audit_points)[:,None],
                                              np.arange(ground_truth_indices.shape[1]), 
                                              ground_truth_indices] for x in sm_shadow_models])
        
        # Converts the token-level logits to document level instead
        p_x_given_shadow_models = [self.token_to_doc_lvl(x, ground_truth_indices) for x in p_x_given_shadow_models]

        # evaluate the marginal p_out(x) by averaging the output of the shadow models
        p_x_out = np.mean(p_x_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_x_given_shadow_models.squeeze()

        # compute the marginal p(x) from P_out and p_in where p_in = a*p_out+b
        p_x = 0.5*((self.offline_a + 1) * p_x_out + (1-self.offline_a))

        # compute the ratio of p(x|theta) to p(x)
        ratio_x = p_x_given_target_model / (p_x + self.epsilon)

        # for each x, compare it with the ratio of all z points
        likelihoods = ratio_x[...,np.newaxis] / self.ratio_z

        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]

        score = np.mean(likelihoods > self.gamma, axis=1)

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)

    def run_attack(self:Self) -> CombinedMetricResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """
        # perform the attack
        if self.online is True:
            self._online_attack()
        else:
            self._offline_attack()

        # create thresholds
        min_signal_val = np.min(np.concatenate([self.in_member_signals, self.out_member_signals]))
        max_signal_val = np.max(np.concatenate([self.in_member_signals, self.out_member_signals]))
        thresholds = np.linspace(min_signal_val, max_signal_val, 1000)


        member_preds = np.greater(self.in_member_signals, thresholds).T
        non_member_preds = np.greater(self.out_member_signals, thresholds).T


        # what does the attack predict on test and train dataset
        predictions = np.concatenate([member_preds, non_member_preds], axis=1)

        # set true labels for being in the training dataset
        true_labels = np.concatenate(
            [
                np.ones(len(self.in_member_signals)),
                np.zeros(len(self.out_member_signals)),
            ]
        )
        signal_values = np.concatenate(
            [self.in_member_signals, self.out_member_signals]
        )

        # generate the PII distribution
        pii_count_tp, pii_count_fp = self.generate_pii_distribution(signal_values, thresholds, member_preds, non_member_preds)
        
        # compute ROC, TP, TN etc
        return CombinedMetricResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signal_values,
            tp_entity_dist = pii_count_tp,
            fp_entity_dist = pii_count_fp,
        )


    def generate_pii_distribution(self:Self, signals, thresholds, member_preds, non_member_preds):

        if self.online is True:
            member_indices = self.audit_data_indices[np.arange(self.n_in_members)]
            non_member_indices = self.audit_data_indices[np.arange(self.n_in_members, self.n_in_members + self.n_out_members)]
        else: 
            member_indices = self.audit_dataset["in_members"]
            non_member_indices = self.audit_dataset["out_members"]
            
        ground_truth_members = self.handler.get_labels(member_indices).astype(int)
        ground_truth_non_members = self.handler.get_labels(non_member_indices).astype(int)

        n_classes = self.handler.target_model.n_classes
        n_in_members = member_indices.shape[0]
        n_out_members = non_member_indices.shape[0]

        pii_occ_of_members = np.zeros((n_in_members, (n_classes-1)//2), dtype=int)
        for i in range(n_in_members):
            # Counts the number of occurences of each class in the ground truth
            try:
                tmp = np.bincount(ground_truth_members[i][ground_truth_members[i]> 0], minlength = n_classes)
            except:
                print("hello there there's been an issue in the rmia again zzz")

            # Combines the different tokens in the same class to one (i.e. removes difference of beginning/insider tokens)
            pii_occ_of_members[i] = np.array([int(tmp[j] + tmp[j+1]) for j in range(1, n_classes-1, 2)])


        pii_occ_of_non_members = np.zeros((n_out_members, (n_classes-1)//2), dtype=int)
        for i in range(n_out_members):
            # Counts the number of occurences of each class in the ground truth
            try: 
                tmp = np.bincount(ground_truth_non_members[i][ground_truth_non_members[i]> 0], minlength = n_classes)
            except:
                print("hello there there's been an issue in the rmia (non members) zzz")
            # Combines the different tokens in the same class to one (i.e. removes difference of beginning/insider tokens)
            pii_occ_of_non_members[i] = np.array([int(tmp[j] + tmp[j+1]) for j in range(1, n_classes-1, 2)])


        n_member_preds = np.sum(member_preds, axis = 1)
        n_non_member_preds = np.sum(non_member_preds, axis = 1)

        total_members = member_preds.shape[1]
        total_non_members = non_member_preds.shape[1]

        # Save the member predictions and non-member predictions
        with open("member_preds.npy", "wb") as f:
            pred_info =  {
                "n_member_preds": n_member_preds,
                "n_non_member_preds": n_non_member_preds,
                "total_members": total_members,
                "total_non_members": total_non_members
                }
            np.save(f, pred_info)
        
        


        pii_dist_tp = np.zeros((len(thresholds), (n_classes-1)//2), dtype=float)
        pii_dist_fp = np.zeros((len(thresholds), (n_classes-1)//2), dtype=float)
        
        for i in range(len(thresholds)):
            # Get the index mask for tp and fp, and the number of them for each threshold
            tp_mask = member_preds[i]
            fp_mask = non_member_preds[i]

            n_true_positives = np.sum(tp_mask)
            n_false_positives = np.sum(fp_mask)
            
            if n_true_positives != 0:
                # Count the number of occurences of each class in the ground truth
                piis_of_tps = pii_occ_of_members[tp_mask]
                binary_pii_tp = (piis_of_tps >1).astype(int)

                pii_dist_tp[i] = np.sum(binary_pii_tp, axis = 0) / binary_pii_tp.sum()
                    
            if n_false_positives != 0:
                # Count the number of occurences of each class in the ground truth
                piis_of_fps = pii_occ_of_non_members[fp_mask]
                binary_pii_fp = (piis_of_fps >1).astype(int)

                pii_dist_fp[i] = np.sum(binary_pii_fp, axis = 0) / binary_pii_fp.sum()

        return pii_dist_tp, pii_dist_fp