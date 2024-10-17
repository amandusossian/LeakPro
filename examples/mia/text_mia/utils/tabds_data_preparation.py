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
from torch import tensor, float32, cuda
from typing import List,Any
from dataclasses import dataclass
from transformers import PreTrainedTokenizerFast


device = 'cuda' if cuda.is_available() else 'cpu'

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
        for _num, (label, s) in enumerate(itertools.product(labels, "BI")):
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
            window_stride=None,
        ):
            self.label_set = label_set
            
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


    def __getitem__(self, idx):
        return self.training_examples[idx]


    def subset(self, indices):

       
        # Create a new Dataset instance to hold the subset
        subset_data = TABDataset.__new__(TABDataset)  # Bypass __init__

        # Directly subset the relevant fields from the original dataset
        subset_data.texts = [self.texts[i] for i in indices]
        subset_data.annotations = [self.annotations[i] for i in indices]
        subset_data.training_examples = [self.training_examples[i] for i in indices]

        # Copy the tokenizer, label_set, tokens_per_batch, and window_stride
        subset_data.tokenizer = self.tokenizer
        subset_data.tokens_per_batch = self.tokens_per_batch
        subset_data.window_stride = self.window_stride
        subset_data.label_set = self.label_set

        return subset_data
        
    
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

def preprocess_tab_dataset(datapath):
    """Get the dataset, download it if necessary, and store it."""
    
    if os.path.exists(datapath + "/tab_train.pkl"):
        with open(datapath+ "/tab_train.pkl", "rb") as f:
            dataset = joblib.load(f)



    else: 
        print(os.path.join(path, "tab_data/tab_train.pkl"))
        assert 1 == 2, "Don't load"
        # TODO: Implement downloader after it is done with the first normal case
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num", 
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
        ]
        
        # Load and clean data
        df_train = pd.read_csv(os.path.join(path, "adult.data"), names=column_names)
        df_test = pd.read_csv(os.path.join(path, "adult.test"), names=column_names, header=0)
        df_test["income"] = df_test["income"].str.replace(".", "", regex=False)
        
        df_concatenated = pd.concat([df_train, df_test], axis=0)
        df_clean = df_concatenated.replace(" ?", np.nan).dropna()

        # Split features and labels
        x, y = df_clean.iloc[:, :-1], df_clean.iloc[:, -1]

        # Categorical and numerical columns
        categorical_features = [col for col in x.columns if x[col].dtype == "object"]
        numerical_features = [col for col in x.columns if x[col].dtype in ["int64", "float64"]]

        # Scaling numerical features
        scaler = StandardScaler()
        x_numerical = pd.DataFrame(scaler.fit_transform(x[numerical_features]), columns=numerical_features, index=x.index)
        
        # Label encode the categories
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        x_categorical_one_hot = one_hot_encoder.fit_transform(x[categorical_features])
        one_hot_feature_names = one_hot_encoder.get_feature_names_out(categorical_features)
        x_categorical_one_hot_df = pd.DataFrame(x_categorical_one_hot, columns=one_hot_feature_names, index=x.index)
        
        # Concatenate the numerical and one-hot encoded categorical features
        x_final = pd.concat([x_numerical, x_categorical_one_hot_df], axis=1)

        # Label encode the target variable
        y = pd.Series(LabelEncoder().fit_transform(y))
        
        # Add numerical features to the dictionary
        dec_to_onehot_mapping = {}
        for i, feature in enumerate(numerical_features):
            dec_to_onehot_mapping[i] = [x_final.columns.get_loc(feature)]  # Mapping to column index

        # Add one-hot encoded features to the dictionary
        for i, categorical_feature in enumerate(categorical_features):
            j = i + len(numerical_features)
            one_hot_columns = [col for col in one_hot_feature_names if col.startswith(categorical_feature)]
            dec_to_onehot_mapping[j] = [x_final.columns.get_loc(col) for col in one_hot_columns]

        #--------------------
        # Create tensor dataset to be stored
        x_tensor = tensor(x_final.values, dtype=float32)
        y_tensor = tensor(y.values, dtype=float32)
        dataset = TABDataset(x_tensor, y_tensor, dec_to_onehot_mapping, one_hot_encoded=True)
        with open(f"{path}/adult_data.pkl", "wb") as file:
            pickle.dump(dataset, file)
            print(f"Save data to {path}.pkl")
    
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
    
    train_loader = DataLoader(train_subset, collate_fn = TrainingBatch, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_subset, collate_fn = TrainingBatch, batch_size=1, shuffle=False)

    return train_loader, test_loader


class TrainingBatch:
    def __getitem__(self, item):
        return getattr(self, item)

    def __init__(self, examples: List[TrainingExample]):
        self.input_ids: torch.Tensor
        self.attention_masks: torch.Tensor
        self.labels: torch.Tensor
        self.identifier_types: List
        self.offsets:List
        input_ids: IntListList = []
        masks: IntListList = []
        labels: IntListList = []
        identifier_types: List = []
        offsets: List = []
        for ex in examples:
            print(dir(ex))
            input_ids.append(ex.input_ids)
            masks.append(ex.attention_masks)
            labels.append(ex.labels)
            identifier_types.append(ex.identifier_types)
            offsets.append(ex.offsets)
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(masks)
        self.labels = torch.LongTensor(labels)
        self.identifier_types = identifier_types
        self.offsets = offsets

        self.input_ids = self.input_ids.to(device)
        self.attention_masks = self.attention_masks.to(device)
        self.labels = self.labels.to(device)

if False:
    tabdata = preprocess_tab_dataset("./")
    train_l, test_l = get_tab_dataloaders(tabdata)

