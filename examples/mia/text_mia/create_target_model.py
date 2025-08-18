import os
import sys
leakpro_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
alvis_root = os.path.abspath(os.path.join(os.getcwd(), "../../../.."))
sys.path.append(leakpro_root)

from examples.mia.text_mia.utils.tabds_data_preparation import *
from examples.mia.text_mia.utils.tabds_model_preparation import *

path = os.path.join(alvis_root, "tab_data/") # /Users/reimera/Desktop/alvis/projects/tab_data/


dataset = preprocess_tab_dataset(path, 
                                create_new = True, 
                                class_masking = True, 
                                dataset_name = "complete")  


print("Finished preprocessing dataset")

n_classes = dataset.label_set.n_classes
train_loader, test_loader = get_tab_dataloaders(dataset, train_fraction=0.4, test_fraction=0.4, batch_size = 2)

n_epochs = 4
# Train the model
if not os.path.exists("target"):
    os.makedirs("target")
model = TABBERT(pt_model= "allenai/longformer-base-4096", n_classes=n_classes)
train_acc, train_loss, test_acc, test_loss = create_trained_model_and_metadata(model, 
                                                                               train_loader, 
                                                                               test_loader, 
                                                                               epochs=n_epochs)

