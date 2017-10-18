import sys
import os
import networkx as nx
import community
from queue import Queue

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(FILE_DIR, '..'))
from utils.env import *
from utils.data_handler import DataHandler as dh
from shared_types import Node

def modify_tree_format(tree, root):
    ret_tree = [None] * len(tree)
    mapp = []
    q = Queue()
    q.put(root)
    while q.empty() == False:
        u = q.get()
        mapp.append(u)
        for v in tree[u].childst:
            q.put(v)

    mapp.reverse()
    r_mapp = {}
    for i in xrange(len(mapp)):
        if len(tree[mapp[i]].childst) == 0:
            for item in tree[mapp[i]].coverst:
                r_mapp[mapp[i]] = item 
        else:
            r_mapp[mapp[i]] = i
    for i in xrange(len(tree)):
        ret_tree[r_mapp[i]] = Node(r_mapp[i], set(), set(tree[i].coverst))
        for v in tree[i].childst:
            ret_tree[r_mapp[i]].childst.add(r_mapp[v])

    return ret_tree


def dfs(u, d, coverst, dep, G, tree):
    node_u = Node(u, set(), coverst)
    tree.append(node_u)
    if d == dep - 1:
        for v in coverst:
            node_u.childst.add(len(tree))
            tree.append(Node(len(tree), set(), set({v})))
        return
    G_now = nx.Graph()
    for v in coverst:
        for e in G[v]:
            if e in coverst:
                G_now.add_edge(v, e)
    partition = community.best_partition(G_now)

    coverst_list = {}
    for key in partition:
        if partition[key] in coverst_list:
            coverst_list[partition[key]].add(key)
        else:
            coverst_list[partition[key]] = set({key})

    for key in coverst_list:
        node_u.childst.add(len(tree))
        dfs(len(tree), d + 1, coverst_list[key], dep, G, tree)

    return tree


def extract_hierarchy(G, logger, params):
    dep = int(params["depth"])
    tree = []
    se = range(G.number_of_nodes())
    dfs(0, 0, se, dep, G, tree)
    tree = modify_tree_format(tree, 0)
    #for i in xrange(len(tree)):
    #    print i
    #    print tree[i]
    return tree
