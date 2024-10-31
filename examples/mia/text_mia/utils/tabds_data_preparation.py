import os
import sys
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import urllib.request
from torch.utils.data import Dataset, Subset, DataLoader
from torch import tensor, float32, cuda, LongTensor, Tensor, long
from typing import List,Any
from dataclasses import dataclass
from transformers import LongformerTokenizerFast, PreTrainedTokenizerFast
from itertools import product
from transformers import BatchEncoding
from tokenizers import Encoding
from torch.nn.utils.rnn import pad_sequence


dev = 'cuda' if cuda.is_available() else 'cpu'

IntList = List[int] # A list of token_ids
IntListList = List[IntList] # A List of List of token_ids, e.g. a Batch

@dataclass
class TrainingExample:
    input_ids: IntList
    attention_masks: IntList
    labels: IntList
    identifier_types: IntList
    offsets:IntList

class LabelSet:
    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0 
        for _num, (label, s) in enumerate(product(labels, "BI")):
            num = _num + 1
            l = f"{s}-{label}"
            self.labels_to_id[l] = num
            self.ids_to_label[num] = l

    def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations, ids):
        raw_labels, identifier_types, offsets, ids = align_tokens_and_annotations_bilou(tokenized_text, annotations, ids)
        return list(map(self.labels_to_id.get, raw_labels)), identifier_types, offsets, ids

class TABDataset(Dataset):

    def __init__(
            self,
            data: Any,
            label_set: LabelSet,
            tokenizer: PreTrainedTokenizerFast,
            tokens_per_batch=32,
            window_stride=None            
        ):
            self.label_set = label_set
            self.tokens_per_batch = tokens_per_batch
            self.window_stride = tokens_per_batch if window_stride is None else window_stride
            self.tokenizer = tokenizer
            self.texts = []
            self.annotations = []
            ids = []

            for example in data:
                self.texts.append(example["text"])
                self.annotations.append(example["annotations"])
                ids.append(example['doc_id'])

            ###TOKENIZE All THE DATA
            tokenized_batch = self.tokenizer(self.texts, add_special_tokens=True, padding = True, return_offsets_mapping=True)

            ## This is used to keep track of the offsets of the tokens, 
            # and used to calculate the offsets on the entity level at evaluation time.
            offset_mapping = []
            for x,y in zip(ids, tokenized_batch.offset_mapping):
                l = []
                for tpl in y:
                    l.append((x, tpl[0], tpl[1]))
                offset_mapping.append(l)

            ###ALIGN LABELS ONE EXAMPLE AT A TIME
            aligned_labels = []
            identifiers = []
            o = []
            for ix in range(len(tokenized_batch.encodings)):
                encoding = tokenized_batch.encodings[ix]
                raw_annotations = self.annotations[ix]
                aligned, identifier_types, outs, ids= label_set.get_aligned_label_ids_from_annotations(
                    encoding, raw_annotations, ids
                )
                aligned_labels.append(aligned)
                identifiers.append(identifier_types)
                o.append(outs)
            ###END OF LABEL ALIGNMENT

            ###MAKE A LIST OF TRAINING EXAMPLES.
            self.training_examples: List[TrainingExample] = []
            empty_label_id = "O"
            for encoding, label, identifier_type, mapping  in zip(tokenized_batch.encodings, aligned_labels, identifiers, offset_mapping):
                length = len(label)  # How long is this sequence
                for start in range(0, length, self.window_stride):
                    end = min(start + tokens_per_batch, length)
                    padding_to_add = 0
                    self.training_examples.append(
                        TrainingExample(
                            # Record the tokens
                            input_ids=encoding.ids[start:end]  # The ids of the tokens
                            + [self.tokenizer.pad_token_id]
                            * padding_to_add,  # padding if needed
                            labels=(
                                label[start:end]
                                + [-1] * padding_to_add  # padding if needed
                            ),
                            attention_masks=(
                                encoding.attention_mask[start:end]
                                + [0]
                                * padding_to_add  # 0'd attention masks where we added padding
                            ),
                            identifier_types=(identifier_type[start:end]
                                + [-1] * padding_to_add ##Not used 
                            
                            ),
                            offsets=(mapping[start:end]
                                + [-1] * padding_to_add
                            ),

                        )
                    )


    def __len__(self):
        return len(self.training_examples)


    def __getitem__(self, idx) -> dict:
        # Retrieve the TrainingExample from the dataset
        ex = self.training_examples[idx]

        # Convert fields to tensors (do the padding if necessary)
        input_ids = tensor(ex.input_ids, dtype=long).to(dev)
        attention_masks = tensor(ex.attention_masks, dtype=long).to(dev)
        labels = tensor(ex.labels, dtype=long).to(dev)
        identifier_types = ex.identifier_types
        offsets= ex.offsets
        # Return dict 
        return {
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'identifier_types': identifier_types, 
            'offsets': offsets  
        }, labels


    def subset(self, indices):

       
        # Create a new Dataset instance to hold the subset
        subset_data = TABDataset.__new__(TABDataset)  # Bypass __init__

        # Directly subset the relevant fields from the original dataset
        
        
        subset_data.training_examples = [self.training_examples[i] for i in indices]
        # Copy the tokenizer, label_set, tokens_per_batch, and window_stride
        subset_data.tokenizer = self.tokenizer
        subset_data.tokens_per_batch = self.tokens_per_batch
        subset_data.window_stride = self.window_stride
        subset_data.label_set = self.label_set

        return subset_data
    
    def collate_fn(self, batch):
        input_ids = [item[0]['input_ids'] for item in batch]
        attention_masks = [item[0]['attention_masks'] for item in batch]
        labels = [item[1] for item in batch]
        
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        
        
        return Batch(input_ids = input_ids, attention_masks = attention_masks), labels
    
class Batch:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to(self, device):
        for key, value in vars(self).items():
            setattr(self, key, value.to(device))
        return self


# TODO: Implement downloader after it is done with the first normal case
'''
def download_tab_dataset(data_dir):
    """Download the Adult Dataset if it's not present."""
    # URLs for the dataset
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    data_file = os.path.join(data_dir, "adult.data")
    test_file = os.path.join(data_dir, "adult.test")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory:", data_dir)
    else:
        print("Directory already exists:", data_dir)

    # Download the dataset if not present
    if not os.path.exists(data_file):
        print("Downloading adult.data...")
        urllib.request.urlretrieve(base_url + "adult.data", data_file)

    if not os.path.exists(test_file):
        print("Downloading adult.test...")
        urllib.request.urlretrieve(base_url + "adult.test", test_file)
'''

def preprocess_tab_dataset(datapath, create_new = False):
    """Get the dataset, download it if necessary, and store it."""
    # maybe can not have file paths in here, but as args instead

    # try to create a new dataset from an existing datafile
    if create_new: 
        print("Creating a dataset from file.")
        if os.path.exists(datapath + "/tab_train_raw_200.pkl"):
            bert = 'allenai/longformer-base-4096'
            label_set = LabelSet(labels=["MASK"])
            with open(datapath+ "/tab_train_raw_200.pkl", "rb") as f:
                dataset  = TABDataset(joblib.load(f), label_set = label_set, 
                                     tokenizer = LongformerTokenizerFast.from_pretrained(bert),
                                     tokens_per_batch=4096) 
            with open(datapath+ "/tab_train_200_dataset.pkl", 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

    # otherwise we try to load a dataset
    else: 
        if os.path.exists(datapath + "/tab_train_200_dataset.pkl"):
            print("Loading local dataset.")
            with open(datapath+ "/tab_train_200_dataset.pkl", "rb") as f:
                dataset = joblib.load(f)


        # otherwise we should download it, but that hasn't been implemented yet
        else: 
            print(os.path.join(datapath, "tab_data/tab_train.pkl"))
            assert 1 == 2, "Can't download datasets yet."
 
    

    return dataset

def get_tab_dataloaders(dataset, train_fraction=0.3, test_fraction=0.3):
    
    dataset_size = len(dataset)
    train_size = int(train_fraction * dataset_size)
    test_size = int(test_fraction * dataset_size)

    # Use sklearn's train_test_split to split into train and test indices
    selected_index = np.random.choice(np.arange(dataset_size), train_size + test_size, replace=False)
    train_indices, test_indices = train_test_split(selected_index, test_size=test_size)
    
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=1, collate_fn=dataset.collate_fn, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=1, collate_fn=dataset.collate_fn, shuffle=False)

    return train_loader, test_loader



class TrainingBatch:
    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, examples: List[TrainingExample]):
        self.input_ids: Tensor
        self.attention_masks: Tensor
        self.labels: Tensor
        self.identifier_types: List
        self.offsets:List
        input_ids: IntListList = []
        masks: IntListList = []
        labels: IntListList = []
        identifier_types: List = []
        offsets: List = []
        for ex in examples:
            
            input_ids.append(ex.input_ids)
            masks.append(ex.attention_masks)
            labels.append(ex.labels)
            identifier_types.append(ex.identifier_types)
            offsets.append(ex.offsets)
        self.input_ids = LongTensor(input_ids)
        self.attention_masks = LongTensor(masks)
        self.labels = LongTensor(labels)
        self.identifier_types = identifier_types
        self.offsets = offsets

        self.input_ids = self.input_ids.to(dev)
        self.attention_masks = self.attention_masks.to(dev)
        self.labels = self.labels.to(dev)

def align_tokens_and_annotations_bilou(tokenized: Encoding, annotations, ids):
    tokens = tokenized.tokens
    identifier_types = ["O"] * len(
        tokens
    )
    aligned_labels = ["O"] * len(
        tokens
    )
    offsets = ["O"] * len(
        tokens
    )
    for anno in annotations:
        ids.append(anno['id'])
        if anno['label'] == 'MASK':
            annotation_token_ix_set = (
                set()
            )  # A set that stores the token indices of the annotation
            for char_ix in range(anno["start_offset"], anno["end_offset"]):

                token_ix = tokenized.char_to_token(char_ix)
                if token_ix is not None:
                    annotation_token_ix_set.add(token_ix)
            last_token_in_anno_ix = len(annotation_token_ix_set) - 1
            for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                if num == 0:
                    prefix = "B"
                else:
                    prefix = "I"  # We're inside of a multi token annotation
                aligned_labels[token_ix] = f"{prefix}-{anno['label']}"
                identifier_types[token_ix] = anno['identifier_type']
                offsets[token_ix] = {anno['id'] : (anno['start_offset'], anno['end_offset'])}

    return aligned_labels, identifier_types, offsets, ids



