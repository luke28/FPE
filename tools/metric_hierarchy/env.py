import os
import sys 

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data")
CONF_PATH = os.path.join(ROOT_PATH, "conf")
RES_PATH = os.path.join(ROOT_PATH, "res")
if os.path.exists(RES_PATH) == False:
    os.mkdir(RES_PATH)
