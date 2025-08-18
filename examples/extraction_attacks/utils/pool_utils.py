import random
import json
import pickle
import os 
from torch import tensor

class PIIPool: 

    def __init__(self, datapath, tokenizer, global_attack_pool_bool = True, attack_pool_type = "full", extension = 0):

        self.base_pool = {}        # The full candidate pool from the dataset
        self.attack_pool = {}           # The attack pool, if different from the full dataset
        self.local_pool = {}            # The agent pools
        self.extension = extension    # Number of entities to add to the attack pool
        self.base_pool_lengths = []
        self.attack_pool_lengths = []
        
        self.global_attack_pool_bool = global_attack_pool_bool
        self.attack_pool_type = attack_pool_type
        self.tokenizer = tokenizer
        self.true_idxs = {}

        if datapath:
            self.create_base_pool(datapath)

    def create_base_pool(self, datapath):
        """
        Create a pool of data for the extraction attack.
        
        datapath: str
            Path to the data file to create a pool from. 
            The candidate pool is a dict with entity types as keys and 
            lists of strings as values.
        """

        #TODO Move to dataset?
        class_masking = True
        if class_masking:
            raw_labels=['PERSON', 'CODE', 'LOC', 'ORG', 'DEM', 'DATETIME', 'QUANTITY', 'MISC']
        else:
            raw_labels = ['MASK']

        conversion_dict = {i: label for i, label in enumerate(raw_labels)}
        conversion_dict[-1] = -1
        reverse_conversion_dict = {v: k for k, v in conversion_dict.items()}
       
        with open(datapath, "r", encoding="utf-8") as f:

            data = json.load(f)
            dct = {}
            for ann_data in data:
                for annotator in ann_data['annotations']:
                    for annotation in ann_data['annotations'][annotator]['entity_mentions']:
                        # Only add entities that are not masked, could be changed to include unmasked, 
                        # to increase the pool size and difficulty
                        if annotation['identifier_type'] != 'NO_MASK': 
                            if annotation['entity_type'] not in dct:
                                dct[annotation['entity_type']] = []
                            dct[annotation['entity_type']].append(tensor(self.tokenizer.encode(annotation["span_text"])[1:-1]))
   
        numeric_dct = {}
        for k, v in dct.items():
            numeric_dct[reverse_conversion_dict[k]] = v
        
        self.base_pool = numeric_dct
        self.base_pool[-1] = [] # Add a list for the "not tagged" entity type, shouldn't need to be used

    def load_attack_pool(self, attack_pool_path):
        """
        Load an attack pool from a file.
        TODO: Consider if this should be added, as there might be not that useful. Maybe load population pool and slice it here instead though.
        attack_pool_path: str
            Path to the file containing the attack pool.
        """
        with open(attack_pool_path, "rb") as f:
            self.attack_pool = pickle.load(f)
            self.attack_pool_lengths = {k: len(v) for k, v in self.attack_pool.items()}

    
    def create_attack_pool(self, target_entities):
        """
        Create a pool of data for the extraction attack.
        
        target_entities: List of tuples
            A list with the token ids and entity types of the entities in
            the target document.
        
        self.attack_pool: Dict
            A dict with entity types as keys and lists of strings as values. What's different between 
            this one and the candidate pool is that the attack pool is smaller, but contains the entities that are
            in the target document. Easier to sample from.
        """
        if self.global_attack_pool_bool:
            self.create_global_pools(target_entities)
        else:
            self.create_local_pools(target_entities)
               


    def create_global_pools(self, target_entities):

        n_target_entities = len(target_entities)
 
        # First enter all of the target entities into the attack pool
        for i in range(n_target_entities):
            entity_type = target_entities[i][1].item()
            import pdb; breakpoint()
            true_entity = target_entities[i][0]
         
            if entity_type > -1:
                if entity_type not in self.attack_pool:
                    self.attack_pool[entity_type] = []
                if true_entity not in self.attack_pool[entity_type]:
                    self.attack_pool[entity_type].append(true_entity)
                    self.true_idxs[i] = len(self.attack_pool[entity_type])-1
                else: 
                    self.true_idxs[i] = self.attack_pool[entity_type].index(true_entity)
                
  
        # Then add some other random entities from the full candidate pool
        for entity_type in self.attack_pool.keys():

            if self.attack_pool_type == "full":
                # Add the full base pool to the attack pool
                self.local_pool[i].append(self.base_pool[entity_type])

            else:
                for i in range(self.extension):
                
                    entity_id = random.randint(0, len(self.base_pool[entity_type]) - 1)
                    self.attack_pool[entity_type].append(self.base_pool[entity_type][entity_id])
                
        

    def create_local_pools(self, target_entities):
        
        n_target_entities = len(target_entities)

        
        # First enter all of the target entities into the attack pool
        for i in range(n_target_entities):
            entity_type = target_entities[i][1]
            true_entity = target_entities[i][0]
         
            if entity_type > -1:
                self.local_pool[i] = []
                self.local_pool[i].append(true_entity)
                self.true_idxs[i] = 0
                
            else: 
                print("Entity type is -1, not adding to agent pool")

            if self.attack_pool_type == "full":
                # Add the full base pool to the attack pool
                self.local_pool[i].append(self.base_pool[entity_type])
            else:
                # Add some other random entities from the base pool
                for j in range(self.extension):
                    entity_id = random.randint(0, len(self.base_pool[entity_type]) - 1)
                    self.local_pool[i].append(self.base_pool[entity_type][entity_id])
  
        


    def sample_from_candidates(self, entity_type, entity_id = -1, local_pool_id = -1):
        """
        Get a sample datapoint for the extraction attack from the candidate pools.
        If provided with entity id, the function will return the entity with that id. 
        """

        # If a local_pool_id is provided, sample from the specified pool
        if local_pool_id != -1:
            if entity_id == -1: 
                entity_id = random.randint(0, len(self.local_pool[local_pool_id]) - 1)
            return self.local_pool[local_pool_id][entity_id]
        
        # Otherwise we samlpe from a global attack pool 
        
        # TODO Check if this line should stay or not
        # entity_type = entity_type.item()
        if self.attack_pool: # If the attack pool is not empty
            if entity_id == -1:
                entity_id = random.randint(0, len(self.attack_pool[entity_type]) - 1)
            return self.attack_pool[entity_type][entity_id]
        
        else:
            if entity_id == -1:
                entity_id = random.randint(0, len(self.base_pool[entity_type]) - 1)
            return self.base_pool[entity_type][entity_id]

    def get_pool(self):
        """
        Return the pool. If an attack pool is present, return the attack pool,
        otherwise the candidate pool.
        """
        if self.local_pool: 
            return self.local_pool
        
        elif self.attack_pool:
            return self.attack_pool
        
        else:
            return self.base_pool
        
    
    def get_pool_sizes(self):
        """ 
        Return the sizes of the pool. If an attack pool is present, return the sizes of the attack pool, 
        otherwise the sizes of the candidate pool. 
        """
        if self.local_pool:
            return {k: len(v) for k, v in self.local_pool.items()}
        elif self.attack_pool:
            return {k: len(v) for k, v in self.attack_pool.items()}
        else:
            return {k: len(v) for k, v in self.base_pool.items()}