import os
import sys
import networkx as nx
import re
import json
import numpy as np
import math


class DataHandler(object):
    @staticmethod
    def load_graph(file_path):
        G = nx.Graph()
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) < 2:
                    continue
                items = line.split()
                G.add_edge(int(items[0]), int(items[1]))
        return G

    @staticmethod
    def transfer_to_matrix(graph):
        n = graph.number_of_nodes()
        mat = np.zeros([n, n])
        for e in graph.edges():
            mat[e[0]][e[1]] = 1
            mat[e[1]][e[0]] = 1
        return mat

    @staticmethod
    def normalize_adj_matrix(g):
        # diagonal should be 1
        mat_ret = g / np.sum(g, axis = 1, keepdims = True)
        return mat_ret

    @staticmethod
    def cal_euclidean_distance(x):
        X = np.array(x)
        a = np.square(np.linalg.norm(X, axis = 1, keepdims = True))
        D = -2 * np.dot(X, np.transpose(X)) + a + np.transpose(a)
        return D

    @staticmethod
    def symlink(src, dst):
        try:
            os.symlink(src, dst)
        except OSError:
            os.remove(dst)
            os.symlink(src, dst)


    @staticmethod
    def load_json_file(file_path):
        with open(file_path, "r") as f:
            s = f.read()
            s = re.sub('\s',"", s)
        return json.loads(s)

    @staticmethod
    def append_to_file(file_path, s):
        with open(file_path, "a") as f:
            f.write(s)


    @staticmethod
    def cal_average_delta(data_set,
                        num_of_nodes,
                        K = 1.0,
                        T = float('inf')):
        n = num_of_nodes
        res = np.zeros((n, n), dtype = float)
        cnt = np.zeros(n, dtype = float)
        for data in data_set:
            for i in xrange(len(data)):
                for j in xrange(i, len(data)):
                    if data[i][1] - data[j][1] > T:
                        break
                    else:
                        res[data[i][0]][data[j][0]] += math.exp(K * (data[i][1] - data[j][1]))
                cnt[data[i][0]] += 1.0
        for i in xrange(n):
            for j in xrange(n):
                if cnt[i] < 0.9:
                    res[i][j] = 0.0
                else:
                    res[i][j] /= cnt[i]
        return res

