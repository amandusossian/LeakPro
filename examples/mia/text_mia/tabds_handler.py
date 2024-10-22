"""Module containing the class to handle the user input for the TAB dataset."""

import torch
from torch import cuda, device, optim, sigmoid
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler

class TABInputHandler(AbstractInputHandler):
    """Class to handle the user input for the TAB dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)


    def get_criterion(self)->None:
        """Set the CrossEntropyLoss for the model."""
        # TODO: Implement the correct loss function for the model
        # Assuming that the output is the probabilities of the different classes, 
        # cross entropy loss feels like a good starting point to investigate.

        if cuda.is_available():
            return CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([1.0, 10.0, 10.0]).cuda())
        else:
            return CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([1.0, 10.0, 10.0]))
        
    def get_optimizer(self, model:torch.nn.Module) -> None:
        """Set the optimizer for the model."""
        # TODO: Evaluate which model optimizer to use, but adam is prolly good
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
        optimizer = self.get_optimizer()
        

        train_acc, train_loss = 0.0, 0.0
        # Training loop
        for e in tqdm(range(epochs), desc="Training Progress"):
            
            model.train()    
            for X in tqdm(dataloader):
                y = X['labels']
                optimizer.zero_grad()
                y_pred = model(X)
                y_pred = y_pred.permute(0,2,1)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_acc += y_pred.eq(y).sum().item()
                train_loss += loss.item()

            print('Epoch', int(e + 1), "done.")
            print('Avg training loss: {0:.2f}'.format(train_loss/(len(dataloader.dataset)*(e+1))))

        train_acc = train_acc/len(dataloader.dataset)
        train_loss = train_loss/len(dataloader)

        return {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}
