from torch.nn import CrossEntropyLoss, Module, Linear
from torch import device, optim, cuda, no_grad, save, sigmoid, Tensor
import pickle
import torch.nn as nn
from transformers import BatchEncoding
from tokenizers import Encoding
from tqdm import tqdm  
import transformers
from sklearn.metrics import recall_score, precision_score
import time 
import numpy as np
import pdb


class TABBERT(nn.Module):
    def __init__(self, pt_model, n_classes):
        # Reworked
        super().__init__()
        self._bert = transformers.LongformerModel.from_pretrained(pt_model)
        self.pt_model = pt_model
        self.n_classes = n_classes

        for param in self._bert.parameters():
           param.requires_grad = True
        
        self.classifier = nn.Linear(768, n_classes)
    
    def forward(self, input_ids, attention_mask):
        # Reworked
        outputs = self._bert( input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)

        return logits

def evaluate(model, loader, criterion, dev):
    # Reworked
    model.eval()
    tot_loss, acc = 0.0, 0.0 

    with no_grad():
        for batch in loader:

            labels = batch["labels"].to(dev)
            input_ids = batch["input_ids"].to(dev)
            attention_masks = batch["attention_masks"].to(dev)
            
            logits = model(input_ids, attention_masks)
            pred_idx = logits.argmax(dim=-1)
            
            val_loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            tot_loss += val_loss.item()
            acc += pred_idx.eq(labels).sum().item()

        n_tokens = logits.shape[1]
        avg_loss = tot_loss / len(loader)
        avg_acc_per_token = float(acc) / ( len(loader.dataset) * n_tokens )

    
    return avg_loss, avg_acc_per_token


def create_trained_model_and_metadata(model, train_loader, test_loader, epochs = 10, metadata = None):
    # Reworked, check the shapes of the input and output of the accuracy again. 
    dev = device("cuda" if cuda.is_available() else "cpu")
   
    model.to(dev)
    model.train()
    
    weight_list = [10.0 for i in range(model.n_classes)] # Assymetric mesasure for weighing the loss function
    weight_list[0] = 1.0

    # Loss function
    if cuda.is_available():
        criterion = CrossEntropyLoss(ignore_index=-1, weight=Tensor(weight_list).cuda())
    else:
        criterion = CrossEntropyLoss(ignore_index=-1, weight=Tensor(weight_list))

    # Optimizer
    optimizer = optim.AdamW(model.parameters(),lr=2e-5, eps=1e-8)

    # Training loop
    train_losses, train_accuracies, train_recalls, train_precisions = [], [], [], []
    test_losses, test_accuracies, test_recalls, test_precisions = [], [], [], []
    print("Creating target model, training started")

    pii_train_accuracies = []
    pii_test_accuracies = []
    # TODO add the code for the pii specific test point. Consider how the doc and batch comes into play here
    
    for e in range(epochs):
        print("Epoch", int(e+1), "started")
        
        model.train()    

        train_acc, train_loss = 0.0, 0.0
        
        current_pii_train_acc = 0.0
        n_seen_pii = 0
        n_masked_tokens = 0

        batch_counter = 0
        n_updates = 5

        for batch in train_loader:

            batch_counter += 1
            if batch_counter % int(len(train_loader) / n_updates )  == 0 and batch_counter != 0:
                print("Batch", str(batch_counter) + "/" + str(len(train_loader)), "completed")
                
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
            
            # Check the accuracies of the actual masks
            for iDoc in range(pred_idx.shape[0]):
                tmp_masked_indices = np.where(labels[iDoc].cpu() != 0)[0] 
                n_masked_tokens += len(tmp_masked_indices)
                n_seen_pii += len(tmp_masked_indices[labels[iDoc][tmp_masked_indices].cpu() % 2 == 1])
                current_pii_train_acc += sum( [1 if x == y else 0 for x, y in zip(labels[iDoc][tmp_masked_indices].cpu(), pred_idx[iDoc][tmp_masked_indices].cpu()) ])
            
            

                    
        n_tokens = logits.shape[1]
        train_loss = train_loss/len(train_loader)
        train_acc = train_acc/ (len(train_loader.dataset) * n_tokens)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        current_pii_train_acc = current_pii_train_acc / n_masked_tokens
        pii_train_accuracies.append(current_pii_train_acc)
        
        print("Evaluating")
        test_loss, test_acc = evaluate(model, test_loader, criterion, dev)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"On epoch {e+1}, there was an accuracy of {current_pii_train_acc} on masked tokens, {n_seen_pii} seen PII in total, overall training accuracy {train_acc}, and test accuracy {test_acc}")
        breakpoint()
    print("Training done!")




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
                                "n_classes": model.n_classes}
    
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
    meta_data["train_acc"] = train_accuracies
    meta_data["test_acc"] = test_accuracies
    meta_data["train_loss"] = train_losses
    meta_data["test_loss"] = test_losses
    meta_data["dataset"] = "tab"
    meta_data["PII_train_acc"] = pii_train_accuracies
    meta_data["PII_test_acc"] = pii_test_accuracies
    
    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)
    
    return train_accuracies, train_losses, test_accuracies, test_losses
