"""Module containing the class to handle the user input for the TAB dataset."""

import torch
from torch import cuda, device, optim, sigmoid
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
# import batch class 
from examples.mia.text_mia.utils.tabds_data_preparation import Batch

from leakpro import AbstractInputHandler

class TABInputHandler(AbstractInputHandler):
    """Class to handle the user input for the TAB dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)


    def get_criterion(self)->None:
        """Set the CrossEntropyLoss for the model."""
        weights = [10.0 for i in range(self.population.n_classes)]  
        weights[0] = 1.0
        if cuda.is_available():
            return CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor(weights).cuda())
        else:
            return CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor(weights))
        
    def get_optimizer(self, model:torch.nn.Module) -> None:
        """Set the optimizer for the model."""
        
        learning_rate = 2e-5
        epsilon = 1e-8
        return AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> dict:
        """Model training procedure."""

        dev = device("cuda" if cuda.is_available() else "cpu")
        
        model.to(dev)
        model.train()

        criterion = self.get_criterion()
        optimizer = self.get_optimizer(model = model)
        

        train_acc, train_loss = 0.0, 0.0
        # Training loop
        for e in tqdm(range(epochs), desc="Training Progress"):
            
            model.train()    
            for X, labels in tqdm(dataloader):
                y = labels.to(dev)
                optimizer.zero_grad()
                
                #X_feat = Batch(input_ids=X['input_ids'], attention_masks=X['attention_masks']).to(dev)
                y_pred = model(X.to(dev))
                y_pred = y_pred.permute(0,2,1)
                pred_idx = y_pred.argmax(dim=1)

                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_acc += pred_idx.eq(y).sum().item()
                train_loss += loss.item()

            print('Epoch', int(e + 1), "done.")
            print('Avg training loss: {0:.2f}'.format(train_loss/(len(dataloader.dataset)*(e+1))))

        train_acc = train_acc/len(dataloader.dataset)
        train_loss = train_loss/len(dataloader)

        return {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}
