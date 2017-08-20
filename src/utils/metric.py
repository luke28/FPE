import os
import sys
import time
import networkx as nx
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from operator import itemgetter

from env import *
from data_handler import DataHandler as dh

class Metric(object):
    @staticmethod
    def cal_metric(TP, FP, TN, FN):
	res = {}
	res["acc"] = float(TP + TN) / float(TP + FP + FN + TN)
	res["precision"] = float(TP) / float(TP + FP) if TP + FP > 0 else 1.0
	res["recall"] = float(TP) / float(TP + FN) if TP + FN > 0 else 1.0
        try:
	    res["F1"] = 1.0 / (1.0 / res["recall"] + 1.0 / res["precision"])
        except ZeroDivisionError:
            res["F1"] = 0.0
	return res

    @staticmethod
    def knn(G_truth, params):
        embeddings = dh.load_json_file(os.path.join(RES_PATH, "TrainRes"))["embeddings"]
        embeddings = np.array(embeddings)
        TP = FP = TN = FN = 0
        tmp = []
        for i in xrange(len(embeddings)):
            tmp.append([])
            for j in xrange(len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                tmp[-1].append((dist, i, j))
            tmp[-1].sort(key = itemgetter(0))
        for item in tmp:
            for it in item[:params["K"]]:
                if it[1] in G_truth and it[2] in G_truth[it[1]]:
                    TP += 1
                else:
                    FP += 1
            for it in item[params["K"]:]:
                if it[1] in G_truth and it[2] in G_truth[it[1]]:
                    FN += 1
                else:
                    TN += 1

        res = Metric.cal_metric(TP, FP, TN, FN)
        print "knn metric:"
        print res
        return res

    @staticmethod
    def coefficient(G_truth, params):
        coefficient = dh.load_json_file(os.path.join(RES_PATH, "TrainRes"))["coefficient"]
        TP = FP = TN = FN = 0
        for i in xrange(len(coefficient)):
            for j in xrange(len(coefficient[i])):
                if abs(coefficient[i][j]) > params["threshold"]:
                    if i in G_truth and j in G_truth[i]:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if i in G_truth and j in G_truth[i]:
                        FN += 1
                    else:
                        TN += 1
        res = Metric.cal_metric(TP, FP, TN, FN)
        print "coefficient metric:"
        print res
        return res

    @staticmethod
    def draw_pr(precision, recall, file_name = "pr.png"):
	index = np.array(range(len(precision)))
        width = 0.3
        tmplist1 = [(x, precision[x]) for x in precision]
        tmplist2 = [(x, recall[x]) for x in recall]
        tmplist1.sort()
        tmplist2.sort()
        X = [x[0] for x in tmplist1]
        y1 = [x[1] for x in tmplist1]
        y2 = [x[1] for x in tmplist2]
        plt.bar(index - width / 2, y2, width, color = "blue", label="recall")
        plt.bar(index + width / 2, y1, width, color = "red", label="precision")
        plt.grid(True, which='major')
        plt.grid(True, which='minor')
        plt.xticks(index, X, rotation = 45, size = 'small')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, ncol=5)

        plt.savefig(file_name)
        plt.close()
