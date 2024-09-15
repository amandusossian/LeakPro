import os
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import urllib.request
from torch.utils.data import Dataset, Subset, DataLoader
from torch import from_numpy


class AdultDataset(Dataset):
    def __init__(self, x, y, ):
        self.x = x 
        self.y = y  
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if isinstance(self.x, (pd.DataFrame, pd.Series)):
            return self.x.iloc[idx], self.y.iloc[idx]
        return self.x[idx], self.y[idx]
    
    
def download_adult_dataset(data_dir):
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

def get_adult_dataset(path):
    """Get the dataset, download it if necessary, and store it."""
    
    # Download the dataset if not present
    download_adult_dataset(path)
    
    if os.path.exists(os.path.join(path, "adult_data.pkl")):
        with open(os.path.join(path, "adult_data.pkl"), "rb") as f:
            dataset = joblib.load(f)
    else: 
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
        x_categorical = x[categorical_features].apply(lambda col: LabelEncoder().fit_transform(col))
        x = pd.concat([x_numerical, x_categorical], axis=1)

        # Label encode the target variable
        y = pd.Series(LabelEncoder().fit_transform(y))
        
        #--------------------
        # Create dataset to be stored
        dataset = AdultDataset(x, y)
        with open(f"{path}/adult_data.pkl", "wb") as file:
            pickle.dump(dataset, file)
            print(f"Save data to {path}.pkl")
    
    return dataset

def collate_fn(batch) -> tuple:
    features_df, labels_df = zip(*batch)
    features_tensor = from_numpy(np.array(features_df, dtype=np.float32))
    labels_tensor = from_numpy(np.array(labels_df, dtype=np.float32))
    return features_tensor, labels_tensor    

def get_adult_dataloaders(dataset, train_fraction=0.3, test_fraction=0.3):
    
    dataset_size = len(dataset)
    train_size = int(train_fraction * dataset_size)
    test_size = int(test_fraction * dataset_size)

    # Use sklearn's train_test_split to split into train and test indices
    selected_index = np.random.choice(np.arange(dataset_size), train_size + test_size, replace=False)
    train_indices, test_indices = train_test_split(selected_index, test_size=test_size)
    
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader
