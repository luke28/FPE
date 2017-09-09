import numpy as np
from sklearn.cluster import KMeans

class ClusteringLib(object):
    @staticmethod
    def kmeans(X, n_clusters, params):
        kmeans = KMeans(n_clusters = n_clusters).fit(X)
        return kmeans.labels_
