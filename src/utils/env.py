import os
import sys

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(FILE_PATH, "../..")
DATA_PATH = os.path.join(ROOT_PATH, "data")
CONF_PATH = os.path.join(ROOT_PATH, "conf")
RES_PATH = os.path.join(ROOT_PATH, "res")
SRC_PATH = os.path.join(ROOT_PATH, "src")
sys.path.insert(0, SRC_PATH)
