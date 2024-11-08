import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)
from examples.mia.text_mia.utils.tabds_data_preparation import *
from examples.mia.text_mia.utils.tabds_model_preparation import *

path_to_datafolder = os.path.join(os.getcwd(), "tab_data/")
dataset = preprocess_tab_dataset(path_to_datafolder, create_new=False)
train_loader, test_loader = get_tab_dataloaders(dataset, train_fraction=0.3, 
test_fraction=0.3)

### Generate target model 

if not os.path.exists("target"):
    os.makedirs("target")



n_classes = 3 # Case dependent, is equal to 2*number of masktypes + 1

pretrained_model_name = "allenai/longformer-base-4096"
model = TABBERT(pt_model= pretrained_model_name, num_classes=n_classes)
n_epochs = 2


train_acc, train_loss, test_acc, test_loss = create_trained_model_and_metadata(model = model, 
                                                                               train_loader = train_loader, 
                                                                               test_loader = test_loader, 
                                                                               epochs = n_epochs)


from tabds_handler import TABInputHandler

from leakpro import LeakPro

# Read the config file
config_path = "audit.yaml"

# Prepare leakpro object
leakpro = LeakPro(TABInputHandler, config_path)

# Run the audit 
leakpro.run_audit()