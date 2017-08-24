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
from operator import itemgetter

from get_network_hierarchy.get_network import GetNetwork as gn
from utils.batch_strategy import BatchStrategy
from utils.env import *
from utils.data_handler import DataHandler as dh
from calculate_euclidean_fractal.cal_fractal import CalFractal as cf
#from utils.metric import Metric

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def dfs(u, tree, handlers, params, res_radius, res_coordinates):
    if len(tree[u].childst) == 0:
        return
    node_in_tree, sim_mat, var_mat = handlers["get_network"].get_network(u, tree)

    print u
    print node_in_tree
    print sim_mat
    print var_mat
    raw_input()

    if (len(node_in_tree) <= 2):
        rc = np.random.random(size = params["embedding_model"]["embedding_size"]) * 2 - 1
        rc_b = rc / np.linalg.norm(rc) * res_radius[u]
        rc = rc_b + res_coordinates[u]
        print(rc)
        res_coordinates[node_in_tree[0]] = rc
        if (len(node_in_tree) == 2):
            rc_b = - rc_b + res_coordinates[u]
            res_coordinates[node_in_tree[1]] = rc_b
            print rc_b
        raw_input()
    elif (len(node_in_tree) == 2):
        rc1 = np.random.random(size = params["embedding_model"]["embedding_size"]) * 2 - 1
        
    else:
        # neural embedding
        sim_mat_norm = dh.normalize_adj_matrix(sim_mat)
        bs = BatchStrategy(sim_mat_norm)
        params["embedding_model"]["num_nodes"] = len(sim_mat_norm)
        ne = handlers["embedding_model"](params["embedding_model"])
        X = ne.train(getattr(bs, params["embedding_model"]["batch_func"]), params["embedding_model"]["iteration"])
        print(X)
        print(dh.cal_euclidean_distance(X))
        raw_input()

        # transfer embedding
        params["transfer_embeddings"]["num_nodes"] = len(sim_mat_norm)

        te = handlers["transfer_embeddings"](params["transfer_embeddings"])
        Z = te.transfer(X, res_coordinates[u], res_radius[u], params["transfer_embeddings"]["iteration"])
        for i in xrange(len(node_in_tree)):
            res_coordinates[node_in_tree[i]] = Z[i]
        raw_input()

    # cal radius
    r = np.zeros(len(node_in_tree), dtype = np.float32)
    cnt = np.zeros(len(r), dtype = np.float32)
    for i in xrange(len(r)):
        for j in xrange(len(r)):
            if sim_mat[i][j] > sys.float_info.epsilon:
                r[i] += var_mat[i][j] / (sim_mat[i][j] * params["scaling_radius"]) * np.linalg.norm(res_coordinates[node_in_tree[i]] - res_coordinates[node_in_tree[j]])
                cnt[i] += 1.0

    for i in xrange(len(r)):
        if cnt[i] > sys.float_info.epsilon:
            r[i] = r[i] / cnt[i]
        res_radius[node_in_tree[i]] = min(params["radius_max"] * res_radius[u], max(params["radius_min"] * res_radius[u], r[i]))

    print res_radius
    print res_coordinates
    raw_input()

    for v in tree[u].childst:
        dfs(v, tree, handlers, params, res_radius, res_coordinates)


def train_model(params, is_save = True):
    g = dh.load_graph(os.path.join(DATA_PATH, params["network_file"]))
    g_mat = dh.transfer_to_matrix(g)
    #print g_mat
    eh = __import__('extract_hierarchy.' + params["extract_hierarchy_model"]["func"], fromlist = ["extract_hierarchy"])
    tree = eh.extract_hierarchy(g_mat, params["extract_hierarchy_model"]["threshold"])

    print [str(i) for i in tree]
    raw_input()
    handlers = {}
    handlers["get_network"] = gn(g_mat, params["get_network_hierarchy"])
    handlers["embedding_model"] = __import__('node_embedding.' + params["embedding_model"]["func"], fromlist = ["node_embedding"]).NodeEmbedding
    handlers["transfer_embeddings"] = __import__('transfer_embeddings.' + params["transfer_embeddings"]["func"], fromlist = ["transfer_embeddings"]).TransferEmbedding

    res_coordinates = [None] * len(tree)
    res_coordinates[len(tree) - 1] = np.zeros(params["embedding_model"]["embedding_size"], dtype = np.float32)
    res_radius = [None] * len(tree)
    res_radius[len(tree) - 1] = float(params["init_radius"])
    dfs(len(tree) - 1, tree, handlers, params, res_radius, res_coordinates)

    print res_radius
    print res_coordinates

    origin_coordinates = res_coordinates[: params["num_nodes"]]
    dim = getattr(cf, params["calculate_euclidean_fractal"]["func"])(origin_coordinates, params["transfer_embeddings"]["embedding_size"], params["calculate_euclidean_fractal"])
    print("dims: ", dim)

'''
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
'''
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
    args = parser.parse_args()
    params = dh.load_json_file(os.path.join(CONF_PATH, args.conf + ".json"))

    if args.operation == "all":
        train_model(params)
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
