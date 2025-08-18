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
        print("TABInputHandler initialized.")


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
        print("Training loop started.")
        for e in range(epochs):
            
            model.train()    
            batch_counter = 0
            n_updates = 5
            for batch in dataloader:

                batch_counter += 1
                if batch_counter % int(len(dataloader) / n_updates )  == 0 and batch_counter != 0:
                    print("Batch", str(batch_counter) + "/" + str(len(dataloader)), "completed")
                    
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(dev)            
                attention_masks = batch['attention_masks'].to(dev)
                labels = batch['labels'].to(dev)

                logits = model(input_ids, attention_masks)

                pred_idx = logits.argmax(dim=-1)
                loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
                loss.backward()
            
                optimizer.step()
            
                train_loss += loss.item()
                # Accuracy calculation
                train_acc += pred_idx.eq(labels).sum().item() 
                

            print('Epoch', int(e + 1), "done.")
            print('Avg training loss: {0:.2f}'.format(train_loss/(len(dataloader.dataset)*(e+1))))
            
        n_tokens = logits.shape[1]
        train_acc = train_acc/(len(dataloader.dataset) * n_tokens) 
        #However, this does not account for the padding tokens. Can add that later.
        train_loss = train_loss/len(dataloader)

        return {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}
