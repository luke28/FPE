#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
import numpy as np
import math

#from extract_hierarchy.ExtractHierarchicalStruc import Node

class GetNetwork(object):
    def __init__(self, adj_mat_, params):
        self.sim_mat_n = eval("GetNetwork."+params['sim_method'])(adj_mat_)
        self.n = len(adj_mat_)

    @staticmethod
    def common_neighbor_sim(adj_mat_):
        n = len(adj_mat_)
        # init diagonal = 1 
        adj_mat = np.zeros([n,n]) + adj_mat_
        for i in xrange(n):
            adj_mat[i][i] = 1
        adj_mat[np.where(adj_mat > 0)] = 1
        sim_mat = np.zeros([n, n])
        for i in xrange(n):
            for j in xrange(i,n):
                if i == j:
                    sim_mat[i][j] = 1
                else:
                    degree = math.sqrt(np.sum(adj_mat[i])*np.sum(adj_mat[j]))
                    comNeighbor = np.sum(adj_mat[i]*adj_mat[j])
                    sim_mat[i][j] = sim_mat[j][i] = comNeighbor*1.0/degree

        return sim_mat

    def get_network(self, fa_id, tree):
        n = self.n 
        childst = list(tree[fa_id].childst)
        n_ch = len(childst)
        #print "childst: \n"
        #print childst
        node_in_tree = []
        for i in xrange(n_ch):
            node_in_tree.append(childst[i])

        # return matrix
        sim_mat = np.zeros([n_ch, n_ch])
        var_mat = np.zeros([n_ch, n_ch])

        for i in xrange(n_ch):
            for j in xrange(i, n_ch):
                if i == j:
                    sim_mat[i][j] = 1
                    var_mat[i][j] = 0
                else:
                    coverst_i = tree[childst[i]].coverst
                    len_i = len(coverst_i)
                    coverst_j = tree[childst[j]].coverst
                    len_j = len(coverst_j)
                    #print "coverst"
                    #print coverst_i
                    #print coverst_j

                    mask_mat = np.zeros([n, n])
                    for p in coverst_i:
                        for q in coverst_j:
                            mask_mat[p][q] = 1

                    mat = mask_mat*self.sim_mat_n
                    # print "mat"
                    # print mat
                    i2j_tmp = np.sum(mat, axis=1)
                    i2j = [i2j_tmp[t]/len_j for t in coverst_i]
                    #print "i2j:"
                    #print i2j
                    sim_mat[i][j] = sim_mat[j][i] = np.mean(i2j)
                    var_mat[i][j] = np.std(i2j) #/math.sqrt(len_i)
                    j2i_tmp = np.sum(mat, axis=0)
                    j2i = [j2i_tmp[t]/len_i for t in coverst_j]
                    #print "j2i:"
                    #print j2i
                    var_mat[j][i] = np.std(j2i) #/math.sqrt(len_j)

        return node_in_tree, sim_mat, var_mat

