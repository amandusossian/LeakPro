import random
import json
import pickle


class PIIPool: 

    def __init__(self, datapath, n_to_add = 0):

        self.candidate_pool = {}        # The full candidate pool from the dataset
        self.create_pii_pool(datapath)
        
        self.attack_pool = {}           # The attack pool, if different from the full dataset
        self.n_to_add = n_to_add        # Number of entities to add to the attack pool
        


        self.candidate_pool_lengths = []
        self.attack_pool_lengths = []

    def create_pii_pool(self, datapath):
        """
        Create a pool of data for the extraction attack.
        
        datapath: str
            Path to the data file to create a pool from. 
            The candidate pool is a dict with entity types as keys and 
            lists of strings as values.
        """
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
                            dct[annotation['entity_type']].append(annotation["span_text"])
                            
        self.candidate_pool = dct
    
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
        
        n_target_entities = len(target_entities)
        
        # First enter all of the target entities into the attack pool
        for i in range(n_target_entities):
            entity_type = target_entities[i][1]
            if entity_type not in self.attack_pool:
                self.attack_pool[entity_type] = []
            self.attack_pool[entity_type].append(target_entities[i][0])

        # Then add some other random entities from the full candidate pool
        for entity_type in self.attack_pool:
            for i in range(self.n_to_add):
                entity_id = random.randint(0, len(self.candidate_pool[entity_type]) - 1)
                self.attack_pool[entity_type].append(self.candidate_pool[entity_type][entity_id])



    def sample_from_candidates(self, entity_type, entity_id = 0):
        """
        Get a sample datapoint for the extraction attack from the candidate pools.
        If provided with entity id, the function will return the entity with that id. 
        """


        if self.attack_pool: # If the attack pool is not empty
            if entity_id == 0:
                entity_id = random.randint(0, len(self.attack_pool[entity_type]) - 1)
            return self.attack_pool[entity_type][entity_id]
        else:
            if entity_id == 0:
                entity_id = random.randint(0, len(self.candidate_pool[entity_type]) - 1)
            return self.candidate_pool[entity_type][entity_id]



    
    

    def get_pool(self):
        """
        Return the pool. If an attack pool is present, return the attack pool,
        otherwise the candidate pool.
        """

        if self.attack_pool:
            return self.attack_pool
        else:
            return self.candidate_pool
    
    def get_pool_sizes(self):
        """ 
        Return the sizes of the pool. If an attack pool is present, return the sizes of the attack pool, 
        otherwise the sizes of the candidate pool. 
        """

        if self.attack_pool:
            return self.attack_pool_lengths
        else:
            return self.candidate_pool_lengths