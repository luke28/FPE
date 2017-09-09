import os
import sys 
import re
import json
import math
import argparse
import time
import subprocess
import numpy as np
import networkx as nx
import tensorflow as tf
import datetime
from operator import itemgetter
from sklearn.preprocessing import scale


from env import *

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.join(os.path.join(FILE_PATH, ".."), ".."), "src"))
#print os.path.join((os.path.join('..', os.path.join('..', FILE_PATH))), "src")
from utils.data_handler import DataHandler as dh
from extract_hierarchy.real_tree import extract_hierarchy as get_tree
from lib_clustering import ClusteringLib
from metric import Metric

def dfs(u, dep, tree, labels_list, n_classes):
    if len(labels_list) == dep:
        return
    n_classes[dep] += len(tree[u].childst)
    for v in tree[u].childst:
        for i in tree[v].coverst:
            labels_list[dep][i] = v
        dfs(v, dep + 1, tree, labels_list, n_classes)

def get_dep(u, dep, tree):
    if len(tree[u].childst) == 0:
        return dep
    ret = dep
    for v in tree[u].childst:
        ret = max(ret, get_dep(v, dep + 1, tree))
    return ret

def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--conf', type = str, default = "default")
    parser.add_argument('--out_name', type = str, default = str(int(time.time() * 1000.0)))
    args = parser.parse_args()
    params = dh.load_json_file(os.path.join(CONF_PATH, args.conf + ".json"))

    m = params["num_nodes"]
    dic = {"file_path": str(r"../tools/metric_hierarchy/data/" + params["tree_file"])}
    tree = get_tree(None, None, dic)
    n = len(tree)
    X = dh.load_fea(os.path.join(DATA_PATH, params["feature_file"]))
    max_dep = get_dep(n - 1, 0, tree)
    labels_list = np.zeros([max_dep - 1, m], dtype = np.int)
    n_classes = np.zeros(max_dep - 1, dtype = np.int)
    dfs(n - 1, 0, tree, labels_list, n_classes)
    print n_classes
    print labels_list
    X_scaled = scale(X)

    out_path = os.path.join(RES_PATH, "res_" + args.out_name)
    f = open(out_path, "w")
    for metric in params["metric_methods"]:
        f.write("method: " + metric + "\n")
        for i in xrange(len(labels_list)):
            pre = getattr(ClusteringLib, params["clustering_model"]["func"])(X_scaled, n_classes[i], params["clustering_model"])
            print pre
            scores = getattr(Metric, metric)(pre, labels_list[i])
            f.write("level: " + str(i) + "\n" + str(scores) + "\n")
    f.close()

if __name__ == "__main__":
    main()
