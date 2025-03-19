import os
import sys
leakpro_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
alvis_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(alvis_root)
sys.path.append(leakpro_root)

import utils.pool_utils
from examples.mia.text_mia.tabds_handler import TABInputHandler
from leakpro.leakpro import LeakPro
config_path = "audit.yaml"

leakpro = LeakPro(TABInputHandler, config_path)
leakpro.run_audit()
