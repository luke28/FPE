#!/urs/bin/env python

from networkx import *
import numpy as np
from random import *
import matplotlib.pyplot as plt
import argparse

def mul(l):
    x=1
    for i in l:
        x = x*i
    return x

def create_tree(l, n, start):
    k = start
    if n == 1:
        return list(range(start, start+l[0]))
    else:
        t = list(range(l[0]))
        for i in range(l[0]):
            t[i] = create_tree(l[1:], n-1, k)
            k = k+mul(l[1:])
        return t

def search_node_from_tree(t, n, node):
    if n==1:
        if node in t:
            return [t.index(node)]
        else:
            return -1
    else:
        for i in range(len(t)):
            tmp = search_node_from_tree(t[i], n-1, node)
            if tmp != -1:
                x=[]
                x.append(i)
                x.extend(tmp)
                return x
        return -1

def tree_to_network(t, l, p_l):
    x = Graph()
    N = mul(l)
    x.add_nodes_from(list(range(N)))
    m = np.zeros([N,N])
    for i in range(N):
        s_i = search_node_from_tree(t, len(l), i)
        for j in range(i+1, N):
            s_j = search_node_from_tree(t, len(l), j)
            distance = len(l)
            for k in range(len(l)):
                if(s_i[k] == s_j[k]):
                    distance = distance-1
                else:
                    break
            m[i][j] = distance
            m[j][i] = distance
            if distance > 0 and uniform(0,1)<p_l[distance]:
                x.add_edge(i, j)
    return x

def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--layers', type = int, nargs='+')
    parser.add_argument('--possib_l', type = float, nargs='+')
    parser.add_argument('--file', type=str, default='sim_net')
    args = parser.parse_args()
    layers = args.layers
    n_layers = len(layers)
    start=0
    p_l = args.possib_l
    file_path = './data/'+args.file
    png_path = './data/'+args.file
    tree = create_tree(layers, n_layers, start)
    net = tree_to_network(tree, layers, p_l)
    nx.draw(net)
    plt.savefig(png_path)

    edges = list(net.edges())
    line1 = str(net.number_of_nodes())+"\t-1\n"
    edges_str = [str(e[0])+"\t"+str(e[1]) for e in edges]

    with open(file_path, 'w') as outfile:
        outfile.write(line1+"\n".join(edges_str))

if __name__=='__main__':
    main()

