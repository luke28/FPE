import os
import sys
import re
import json
import math
import argparse
import time
import numpy as np
import networkx as nx
import tensorflow as tf
from operator import itemgetter

from batch_strategy import BatchStrategy
from utils.env import *
from utils.data_handler import DataHandler as dh
from utils.metric import Metric

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def train_model(params, is_save = True):
    bs = BatchStrategy(params)
    module = __import__(params["model"]).NodeSkipGram
    nkg = module(params)
    embeddings, coefficient = nkg.Train(getattr(bs, params["batch_func"]), epoch_num = params["iteration"])
    if is_save:
        d = {"embeddings": embeddings.tolist(), "coefficient": coefficient.tolist()}
        file_path = os.path.join(RES_PATH, "training_res_" + str(int(time.time() * 1000.0)))
        with open(file_path, "w") as f:
            f.write(json.dumps(d))
        try:
            os.symlink(file_path, os.path.join(RES_PATH, "TrainRes"))
        except OSError:
            os.remove(os.path.join(RES_PATH, "TrainRes"))
            os.symlink(file_path, os.path.join(RES_PATH, "TrainRes"))

def metric(params):
    G_truth = dh.load_ground_truth(os.path.join(DATA_PATH, params["ground_truth_file"]))
    ret = []
    for metric in params["metric_function"]:
        ret.append(getattr(Metric, metric["func"])(G_truth, metric))
    return ret


def main():
    parser = argparse.ArgumentParser(
                formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--operation', type = str, default = "all", help = "[all | train | metric | draw]")
    parser.add_argument('--conf', type = str, default = "default")
    parser.add_argument('--iteration', type = int, default = 10001)
    parser.add_argument('--model', type = str, default = "model_simple")
    args = parser.parse_args()
    params = dh.load_json_file(os.path.join(SINGLE_CONF_PATH, args.conf + ".json"))
    params["iteration"] = args.iteration
    params["model"] = args.model

    if args.operation == "all":
        train_model(params)
        metric(params)
    elif args.operation == "train":
        train_model(params)
    elif args.operation == "metric":
        metric(params)
    elif args.operation == "draw":
        pass
    else:
        print "Not Support!"

if __name__ == "__main__":
    main()
