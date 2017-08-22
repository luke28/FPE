import sys
import os
import re
import math
import numpy as np

class BatchStrategy(object):
    def __init__(self, graph_mat):
        self.graph = graph_mat
        self.n = 0
        self.num_nodes = len(graph_mat)


    def sequential_weighted(self, batch_size):
        batch_x = []
        batch_y = []
        for _ in xrange(batch_size):
            batch_x.append([self.n])
            batch_y.append(self.graph[self.n])
            self.n = (self.n + 1) % self.num_nodes
        return batch_x, batch_y

