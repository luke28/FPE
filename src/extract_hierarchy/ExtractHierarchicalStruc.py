#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
import numpy as np
import math

class Node:
    def __init__(self, id, childst, coverst):
        self.id = id
        self.childst = childst
        self.coverst = coverst

    def __str__(self):
        line = "id: %s\n" % self.id
        line += "childst: %s\n" % self.childst
        line += "coverst: %s\n" % self.coverst
        return line

class UnionFind:
    def __init__(self, n):
        self.father = range(n) # initial the father pointer array 
        self.sets = {}
        for i in self.father:
            self.sets[i] = set([i])

    def find(self, i):
        if i == self.father[i]:
            return i
        self.father[i] = self.find(self.father[i])
        return self.father[i]

    def union(self, i, j):
        fa_i = self.find(i)
        fa_j = self.find(j)
        if fa_i != fa_j:
            self.father[fa_j] = fa_i
            # union sets
            newset = self.sets[fa_i] | self.sets[fa_j]
            del self.sets[fa_i]
            del self.sets[fa_j]
            self.sets[self.father[fa_j]] = newset
        
def CalSimMatrix(adj_mat_):
    n = len(adj_mat_)
    # init diagonal = 1
    adj_mat = np.zeros(n*n).reshape(n, n)
    adj_mat = adj_mat+adj_mat_
    for i in xrange(n):
        adj_mat[i][i] = 1

    simMat = np.zeros(n*n).reshape(n, n)
    adj_mat[np.where(adj_mat > 0)] = 1
    for i in xrange(n):
        for j in xrange(i, n):
            if i == j:
                simMat[i][j] = 1
                continue
            #degree = min(np.sum(adj_mat[i]), np.sum(adj_mat[j]))
            degree = math.sqrt(np.sum(adj_mat[i])*np.sum(adj_mat[j]))
            comNeighbor = np.sum(adj_mat[i]*adj_mat[j])
            simMat[i][j] = simMat[j][i] = comNeighbor*1.0/degree

    return simMat

def GetEdges(sim_mat):
    lst_edge = []
    n = len(sim_mat)
    for i in xrange(n):
        for j in xrange(i+1, n):
            if sim_mat[i][j] != 0:
                lst_edge.append((i, j, sim_mat[i][j]))
    lst_sorted = sorted(lst_edge, key=lambda e:e[2], reverse=1)
    return lst_sorted

def ExtractHStruc(n, lst_edge, thres):
    # init Union Find
    uf = UnionFind(n)
    # init Tree 
    tree = []
    for i in range(n):
        tree.append(Node(i, set([i]), set([i])))
    n_tr = n

    # init the weight of edge of latest snapshot
    pre_w = 0
    for e in lst_edge:
        if uf.find(e[0]) != uf.find(e[1]):
            # whether to snapshot the current union_find status
            # print str(pre_w-e[2])
            if pre_w != 0 and pre_w - e[2] > thres:
                # convert every sets of union find to node and reset the set of union find
                # print "layer: \n"
                for key in uf.sets:
                    childst = uf.sets[key]
                    coverst = set()
                    for s in childst:
                        coverst = coverst | tree[s].coverst
                    tree.append(Node(n_tr, childst, coverst))
                    #print tree[n_tr]
                    uf.sets[key] = set([n_tr])
                    n_tr += 1
            pre_w = e[2]
            # union the two sets
            uf.union(e[0], e[1])

    # generate the root node of tree
    if len(uf.sets) == 1:
        return tree

    childst = set()
    for key in uf.sets:
        childst = childst | uf.sets[key]
    coverst = set()
    for s in childst:
        coverst = coverst | tree[s].coverst

    tree.append(Node(n_tr, childst, coverst))

    return tree

# input: adj_matrix(numpy), thres
# output: nodes of tree
def extract_hierarchy(adj_matrix, thres):
    # calculate the similarity matrix of adj_matrix
    sim_matrix = CalSimMatrix(adj_matrix)

    # statistic and sort edges
    lst_edge = GetEdges(sim_matrix)

    # extract the hierarchical structure
    n = len(adj_matrix)
    tree = ExtractHStruc(n, lst_edge, thres)

    return tree

def test():
    #t = np.array([[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,1,0,0,0],[1,1,1,1,1,0,0,0],[0,0,1,1,1,0,0,1],[0,0,0,0,0,1,1,1],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,1]])
    t = np.array([[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]])
    sim_matrix = CalSimMatrix(t)
    print "sim_matrix: \n"
    print sim_matrix
    lst_edge = GetEdges(sim_matrix)
    print "lst_edges: \n"
    print lst_edge
    n = len(sim_matrix)
    tree = ExtractHStruc(n, lst_edge, 0.08)
    print [str(i) for i in tree]

if __name__ == '__main__':
    test()

