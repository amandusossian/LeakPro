from torch.nn import CrossEntropyLoss, Module, Linear
from torch import device, optim, cuda, no_grad, save, sigmoid, Tensor
import pickle
import torch.nn as nn
from transformers import BatchEncoding
from tokenizers import Encoding
from tqdm import tqdm  
import transformers

class TABBERT(nn.Module):
    def __init__(self, pt_model, num_classes):
        
        super().__init__()
        self._bert = transformers.LongformerModel.from_pretrained(pt_model)
        self.pt_model = pt_model
        self.num_classes = num_classes

        for param in self._bert.parameters():
           param.requires_grad = True
        
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, batch):
        dev = device("cuda" if cuda.is_available() else "cpu")
        b = self._bert(
            input_ids=batch["input_ids"], attention_mask=batch["attention_masks"]
        )
        pooler = b.last_hidden_state
        return self.classifier(pooler)

def evaluate(model, loader, criterion, dev):
    model.eval()
    loss, acc = 0, 0
    
    with no_grad():
        for X in tqdm(loader):
            y = X['labels']
            y_pred = model(X)
            y_pred = y_pred.permute(0,2,1)
            val_loss = criterion(y_pred, y)
            loss += val_loss.item()
            acc += y_pred.eq(y).sum().item()
        loss /= len(loader)
        acc = float(acc) / len(loader.dataset)
    
    return loss, acc

def create_trained_model_and_metadata(model, train_loader, test_loader, epochs = 10, metadata = None):

    dev = device("cuda" if cuda.is_available() else "cpu")
   
    model.to(dev)
    model.train()

    criterion = None
    if cuda.is_available():
       
        criterion = CrossEntropyLoss(ignore_index=-1, weight=Tensor([1.0, 10.0, 10.0]).cuda())
    else:
       
        criterion = CrossEntropyLoss(ignore_index=-1, weight=Tensor([1.0, 10.0, 10.0]))

    optimizer = optim.AdamW(model.parameters(),lr=2e-5, eps=1e-8)
    
    
    

    # Training loop
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    print("Training")
    for e in range(epochs):
        print("Epoch", int(e+1), "started")
        model.train()    

        train_acc, train_loss = 0.0, 0.0

        for X in tqdm(train_loader):
          
            y = X['labels']
            y_pred = model(X)
            optimizer.zero_grad()
            y_pred = y_pred.permute(0,2,1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_acc += y_pred.eq(y).sum().item()
            train_loss += loss.item()

        train_loss = train_loss/len(train_loader)
        train_acc = train_acc/len(train_loader.dataset)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        print("Evaluating")
        test_loss, test_acc = evaluate(model, test_loader, criterion, dev)
        
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)



    # Move the model back to the CPU
    model.to("cpu")
    with open("target/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

    # Create metadata and store it
    meta_data = {}
    meta_data["train_indices"] = train_loader.dataset.indices
    meta_data["test_indices"] = test_loader.dataset.indices
    meta_data["num_train"] = len(meta_data["train_indices"])
    
    # Pre-trained model name and number of classes
    meta_data["init_params"] = {"pt_model": model.pt_model,
                                "num_classes": model.num_classes}
    
    # read out optimizer parameters
    meta_data["optimizer"] = {}
    meta_data["optimizer"]["name"] = optimizer.__class__.__name__.lower()
    meta_data["optimizer"]["lr"] = optimizer.param_groups[0].get("lr", 0)
    meta_data["optimizer"]["weight_decay"] = optimizer.param_groups[0].get("weight_decay", 0)
    meta_data["optimizer"]["momentum"] = optimizer.param_groups[0].get("momentum", 0)
    meta_data["optimizer"]["dampening"] = optimizer.param_groups[0].get("dampening", 0)
    meta_data["optimizer"]["nesterov"] = optimizer.param_groups[0].get("nesterov", False)

    # read out criterion parameters
    meta_data["loss"] = {}
    meta_data["loss"]["name"] = criterion.__class__.__name__.lower()

    meta_data["batch_size"] = train_loader.batch_size
    meta_data["epochs"] = epochs
    meta_data["train_acc"] = train_acc
    meta_data["test_acc"] = test_acc
    meta_data["train_loss"] = train_loss
    meta_data["test_loss"] = test_loss
    meta_data["dataset"] = "tab"
    
    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)
    
    return train_accuracies, train_losses, test_accuracies, test_losses
