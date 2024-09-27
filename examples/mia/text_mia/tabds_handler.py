"""Module containing the class to handle the user input for the TAB dataset."""

import torch
from torch import cuda, device, optim, sigmoid
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
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
        # Assuming that the output is the probabilities of the differenent classes, 
        # cross entropy loss feels like a good starting point to investigate.
        # Later on, maybe look at BCE if we say adversary only has access to masking decisions.
        return CrossEntropyLoss()

    def get_optimizer(self, model:torch.nn.Module) -> None:
        """Set the optimizer for the model."""
        # TODO: Evaluate which model optimizer to use, but adam is prolly good
        learning_rate = 0.1
        momentum = 0.8
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

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
        optimizer = self.get_optimizer(model)
        
        for e in tqdm(range(epochs), desc="Training Progress"):
            model.train() # Why is this here? We're not changing it in the loop
            train_acc, train_loss = 0.0, 0.0
            
            for data, target in dataloader:
                target = target.float().unsqueeze(1)
                data, target = data.to(dev, non_blocking=True), target.to(dev, non_blocking=True)
                optimizer.zero_grad()
                output = model(data)

                loss = criterion(output, target)
                
                
                # TODO: Adjust prediction, based on classification rather than binary?
                # done?

                # For adult dataset, they select the class with the highest probability as the prediction
                # pred = sigmoid(output) >= 0.5
                
                # For the TAB text dataset, our prediction should/could be the class/entity
                # with the highest probability insted (i.e. the argmax of the output.)
                pred = output.argmax(dim=1)
                

                train_acc += pred.eq(target).sum().item()
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        train_acc = train_acc/len(dataloader.dataset)
        train_loss = train_loss/len(dataloader)

        return {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}
