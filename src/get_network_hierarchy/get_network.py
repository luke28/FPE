#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
import numpy as np
import math

#from extract_hierarchy.ExtractHierarchicalStruc import Node

class GetNetwork(object):
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


    @staticmethod
    def get_network(fa_id, tree, adj_mat_, params):
        # init 01 adj_mat
        n = len(adj_mat_)
        adj_mat = np.zeros([n,n])+adj_mat_
        # set diagonal elements
        for i in xrange(n):
            adj_mat[i][i] = 1
        adj_mat[np.where(adj_mat > 0)] = 1
        #print params['sim_method']
        sim_mat_n = eval("GetNetwork."+params['sim_method'])(adj_mat)

        #print "sim_mat_n\n"
        #print sim_mat_n
        
        childst = list(tree[fa_id].childst)
        n_ch = len(childst)
        #print "childst: \n"
        #print childst

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

                    mat = mask_mat*sim_mat_n
                    #print "mat"
                    #print mat
                    i2j_tmp = np.sum(mat, axis=1)
                    i2j = [i2j_tmp[t]/len_j for t in coverst_i]
                    #print "i2j:"
                    #print i2j
                    sim_mat[i][j] = sim_mat[j][i] = np.mean(i2j)
                    var_mat[i][j] = np.std(i2j)
                    j2i_tmp = np.sum(mat, axis=0)
                    j2i = [j2i_tmp[t]/len_i for t in coverst_j]
                    #print "j2i:"
                    #print j2i
                    var_mat[j][i] = np.std(j2i)

        return sim_mat, var_mat

