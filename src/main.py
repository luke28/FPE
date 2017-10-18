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

from get_network_hierarchy.get_network import GetNetwork as gn
from utils.batch_strategy import BatchStrategy
from utils.env import *
from utils.data_handler import DataHandler as dh
from utils import log
from utils.metric import Metric

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def dfs(u, tree, handlers, params, logger, res_radius, res_coordinates):
    if len(tree[u].childst) == 0:
        return
    logger.info("node id: " + str(u))

    starttime = datetime.datetime.now()
    node_in_tree, sim_mat, var_mat = handlers["get_network"].get_network(u, tree)

    endtime = datetime.datetime.now()
    logger.info("get_network time: " + str((endtime - starttime).seconds))
    logger.info("size of child set: " + str(len(node_in_tree)))

    #logger.debug("node_in_tree\n" + str(node_in_tree))
    #logger.debug("sim matrix: \n" + str(sim_mat))
    #logger.debug("var matrix: \n" + str(var_mat))

    if (len(node_in_tree) <= 2):
        #logger.debug("size of child set is less than 2")
        rc = np.random.random(size = params["embedding_model"]["embedding_size"]) * 2 - 1
        rc_b = rc / np.linalg.norm(rc) * res_radius[u]
        rc = rc_b + res_coordinates[u]
        res_coordinates[node_in_tree[0]] = rc
        if (len(node_in_tree) == 2):
            rc_b = - rc_b + res_coordinates[u]
            res_coordinates[node_in_tree[1]] = rc_b
        #logger.debug("coordinates of nodes: ")
        #for i in node_in_tree:
            #logger.debug("\n" + str(res_coordinates[i]))
    else:
        # neural embedding
        starttime = datetime.datetime.now()
        sim_mat_norm = dh.normalize_adj_matrix(sim_mat)
        if "batch_params" in params:
            bs = BatchStrategy(sim_mat_norm, params["batch_params"])
        else:
            bs = BatchStrategy(sim_mat_norm, {})
        params["embedding_model"]["num_nodes"] = len(sim_mat_norm)
        ne = handlers["embedding_model"](params["embedding_model"])
        X = ne.train(getattr(bs, params["embedding_model"]["batch_func"]), params["embedding_model"]["iteration"])

        endtime = datetime.datetime.now()
        logger.info("neural embedding time: " + str((endtime - starttime).seconds))
        #logger.debug("neural embedding:")
        #logger.debug("embedding res: \n" + str(X))
        #logger.debug("distance: \n" + str(dh.cal_euclidean_distance(X)))
        del ne, bs

        # transfer embedding
        starttime = datetime.datetime.now()
        params["transfer_embeddings"]["num_nodes"] = len(sim_mat_norm)

        te = handlers["transfer_embeddings"](params["transfer_embeddings"])
        Z, dic = te.transfer(X, res_coordinates[u], res_radius[u], params["transfer_embeddings"]["iteration"])
        for i in xrange(len(node_in_tree)):
            res_coordinates[node_in_tree[i]] = Z[i]

        endtime = datetime.datetime.now()
        logger.info("transfer time: " + str((endtime - starttime).seconds))
        #logger.debug("transfer embedding:")
        #logger.debug("embedding res: \n" + str(Z))
        #logger.debug("distance: \n" + str(dic))
        
        del te, dic, sim_mat_norm

    starttime = datetime.datetime.now()
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

    del r, cnt, node_in_tree, sim_mat, var_mat

    endtime = datetime.datetime.now()
    logger.info("cal_radius time: " + str((endtime - starttime).seconds))

    #logger.debug("Finish cal radius")
    #logger.debug("radius set:\n" + str(res_radius))
    #logger.debug("res_coordinates:\n" + str(res_coordinates))

    del starttime, endtime
    for v in tree[u].childst:
        dfs(v, tree, handlers, params, logger, res_radius, res_coordinates)


def train_model(params, logger):
    g_mat, tree = extract_tree(params, logger)

    handlers = {}
    handlers["get_network"] = gn(g_mat, params["get_network_hierarchy"])
    handlers["embedding_model"] = __import__('node_embedding.' + params["embedding_model"]["func"], fromlist = ["node_embedding"]).NodeEmbedding
    handlers["transfer_embeddings"] = __import__('transfer_embeddings.' + params["transfer_embeddings"]["func"], fromlist = ["transfer_embeddings"]).TransferEmbedding

    res_coordinates = [None] * len(tree)
    res_coordinates[len(tree) - 1] = np.zeros(params["embedding_model"]["embedding_size"], dtype = np.float32)
    res_radius = [None] * len(tree)
    res_radius[len(tree) - 1] = float(params["init_radius"])
    dfs(len(tree) - 1, tree, handlers, params, logger, res_radius, res_coordinates)

    #logger.debug("final result of radius: \n" + str(res_radius))
    #logger.debug("final result of coordinates: \n" + str(res_coordinates))

    res_path = params["train_output"]
    dh.symlink(res_path, os.path.join(RES_PATH, "new_train_res"))
    dh.append_to_file(res_path, json.dumps({"radius": np.array(res_radius).tolist(), "coordinates": np.array(res_coordinates).tolist()}))

    return res_coordinates, res_radius

def extract_tree(params, logger):
    g = dh.load_graph(os.path.join(DATA_PATH, params["network_file"]))
    g_mat = dh.transfer_to_matrix(g)
    eh = __import__('extract_hierarchy.' + params["extract_hierarchy_model"]["func"], fromlist = ["extract_hierarchy"])
    tree = eh.extract_hierarchy(g, logger, params["extract_hierarchy_model"], )

    logger.info("constuct a tree: \n")
    logger.info("\n" + log.serialize_tree_level(tree))
    return g_mat, tree


def metric(params):
    js = json.loads(open(params["metric_input"]).read())
    coordinates = np.array(js["coordinates"])
    radius = np.array(js["radius"])
    res_path = params["metric_output"]
    dh.symlink(res_path, os.path.join(RES_PATH, "new_metric_res"))
    ret = []
    for metric in params["metric_function"]:
        if metric["metric_func"] == "draw_circle_2D":
            pic_path = os.path.join(PIC_PATH, "draw_circle_" + str(int(time.time() * 1000.0)) + ".pdf")
            dh.symlink(pic_path, os.path.join(PIC_PATH, "new_draw_circle"))
            getattr(Metric, metric["metric_func"])(coordinates, radius, metric, params["num_nodes"], pic_path)
        else:
            origin_coordinates = coordinates[: params["num_nodes"]]
            res = getattr(Metric, metric["metric_func"])(origin_coordinates, metric)
            ret.append((metric["metric_func"], res))
    dh.append_to_file(res_path, json.dumps(ret))

    return ret


def main():

    parser = argparse.ArgumentParser(
                formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--log_print', type = str, default = "no")
    parser.add_argument('--operation', type = str, default = "all", help = "[all | extract_tree | train | metric | draw]")
    parser.add_argument('--conf', type = str, default = "default")
    parser.add_argument('--metric_input', type = str, default = "new_train_res")
    parser.add_argument('--train_output', type = str, default = str(int(time.time() * 1000.0)))
    parser.add_argument('--metric_output', type = str, default = str(int(time.time() * 1000.0)))

    args = parser.parse_args()
    params = dh.load_json_file(os.path.join(CONF_PATH, args.conf + ".json"))
    params["metric_input"] = os.path.join(RES_PATH, args.metric_input)
    params["train_output"] = os.path.join(RES_PATH, "train_res_" + args.train_output)
    params["metric_output"] = os.path.join(RES_PATH, "metric_res_" + args.metric_output)

    if args.log_print == "yes":
        logger = log.get_logger()
    else:
        log_path = os.path.join(LOG_PATH, str(int(time.time() * 1000.0)) + ".log")
        logger = log.get_logger(log_path)

    if args.operation == "all":
        train_model(params, logger)
        metric(params)
    elif args.operation == "extract_tree":
        extract_tree(params, logger)
    elif args.operation == "train":
        train_model(params, logger)
    elif args.operation == "metric":
        metric(params)
    elif args.operation == "draw":
        pass
    else:
        print "Not Support!"
        #logger.debug("operation is not supported")
    if args.log_print == "no":
        dh.symlink(log_path, os.path.join(LOG_PATH, "new_log"))

if __name__ == "__main__":
    main()
