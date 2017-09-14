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


FILE_PATH = os.path.dirname(os.path.abspath(__file__))



def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--input_file', type = str, required = True)
    parser.add_argument('--out_file', type = str, default = str(int(time.time() * 1000.0)))
    args = parser.parse_args()
    args.input_file = os.path.join(FILE_PATH, args.input_file)
    args.out_file = os.path.join(FILE_PATH, "res_" + args.out_file)

    T = nx.DiGraph()
    class_dic = {}
    n = 0
    with open(args.input_file, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if items[1] not in class_dic:
                class_dic[items[1]] = [int(items[0])]
            else:
                class_dic[items[1]].append(int(items[0]))
            n += 1
    m = n
    root = n + len(class_dic)
    for key in class_dic:
        print key
        for item in class_dic[key]:
            T.add_edge(m, item)
        T.add_edge(root, m)
        m += 1
    m = root + 1
    with open(args.out_file, "w") as f:
        f.write(str(m) + "\t" + str(n) + "\n")
        for u, v in T.edges():
            f.write(str(u) + "\t" + str(v) + "\n")



if __name__ == "__main__":
    main()
