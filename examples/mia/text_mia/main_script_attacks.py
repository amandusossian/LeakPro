import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)
from examples.mia.text_mia.utils.tabds_data_preparation import *
from examples.mia.text_mia.utils.tabds_model_preparation import *



from tabds_handler import TABInputHandler

from leakpro import LeakPro

# path = os.path.join(os.getcwd(), "tab_data/")
# dataset = preprocess_tab_dataset(path, 
#                                 create_new = True, 
#                                 class_masking = True, 
#                                 dataset_name = "complete")  
# n_classes = dataset.label_set.n_classes
# train_loader, test_loader = get_tab_dataloaders(dataset, train_fraction=0., test_fraction=0.4)
# n_epochs = 2
# # Train the model
# if not os.path.exists("target"):
#     os.makedirs("target")
# model = TABBERT(pt_model= "allenai/longformer-base-4096", n_classes=n_classes)
# train_acc, train_loss, test_acc, test_loss = create_trained_model_and_metadata(model, 
#                                                                                train_loader, 
#                                                                                test_loader, 
#                                                                                epochs=n_epochs)


# Read the config file
config_path = "audit.yaml"

# Prepare leakpro object
leakpro = LeakPro(TABInputHandler, config_path)

# Run the audit 
leakpro.run_audit()