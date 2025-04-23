# # kmeans.py using any of the 20-odd metrics in scipy.spatial.distance
# # kmeanssample 2 pass, first sample sqrt(N)

# from __future__ import division

# import random

# import numpy as np
# import torch
# from scipy.sparse import issparse  # $scipy/sparse/csr.py
# from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py

# # http://docs.scipy.org/doc/scipy/reference/spatial.html

# # X sparse, any cdist metric: real app ?
# # centres get dense rapidly, metrics in high dim hit distance whiteout
# # vs unsupervised / semi-supervised svm


# class KMeans:

#     def __init__(self, num_cluster, metrics, max_iter=100):
#         self.num_cluster = num_cluster
#         self.metrics = metrics
#         self.cluster_centers_ = None
#         self.max_iter = max_iter

#     def init_cluster_centers_(self, X):
#         n_sample, _ = X.shape
#         return X[np.random.choice(n_sample, self.num_cluster, replace=False)]

#     def calculate_distance(self, X, cluster_centers=None):
#         if cluster_centers is None:
#             cluster_centers = self.cluster_centers_
#         if self.metrics == "torus_euclidean":
#             distance_1 = cdist(X % (2 * np.pi), cluster_centers, "euclidean")
#             distance_2 = cdist(X % (2 * np.pi) - np.pi, cluster_centers, "euclidean")
#             # print(X.shape)
#             # print(distance_1.shape)
#             # print(distance_2.shape)
#             distance = np.where(distance_1 < distance_2, distance_1, distance_2)
#         else:
#             distance = cdist(X, cluster_centers, self.metrics)

#         return distance

#     def fit(self, X, epsilon=1e-3):
#         if torch.is_tensor(X):
#             X = X.numpy()
#         self.cluster_centers_ = self.init_cluster_centers_(X)
#         new_distance = self.calculate_distance(X)
#         ld_cluster_centers_ = np.full_like(self.cluster_centers_, float("inf"))
#         ld_old_cluster_centers_ = np.full_like(self.cluster_centers_, float("inf"))
#         it = 0
#         while (
#             np.abs(ld_cluster_centers_ - self.cluster_centers_).sum() > epsilon
#             and it < self.max_iter
#             and np.abs(ld_old_cluster_centers_ - self.cluster_centers_).sum() > epsilon
#         ):

#             ld_old_cluster_centers_ = ld_cluster_centers_
#             ld_cluster_centers_ = self.cluster_centers_
#             it += 1
#             new_distance = self.calculate_distance(X)
#             # print("NEW DISTANCE", new_distance.shape)
#             # print("cluster_centers_", self.cluster_centers_.shape)
#             dependency = np.argmin(new_distance, axis=1)
#             for i in range(self.num_cluster):
#                 clusters_dependency_1 = X[dependency == i] % (2 * np.pi)
#                 clusters_dependency_2 = X[dependency == i] % (2 * np.pi) - np.pi
#                 mean_1 = np.mean(clusters_dependency_1, axis=0)
#                 mean_2 = np.mean(clusters_dependency_2, axis=0)
#                 dist_1 = self.calculate_distance(
#                     clusters_dependency_1, np.array([mean_1])
#                 )
#                 dist_2 = self.calculate_distance(
#                     clusters_dependency_2, np.array([mean_2])
#                 )
#                 if dist_1.sum() < dist_2.sum():
#                     self.cluster_centers_[i] = mean_1
#                 else:
#                     self.cluster_centers_[i] = mean_2

#                 # self.cluster_centers_[i] = np.mean(X[dependency == i], axis=0)

#     def predict(
#         self,
#         X,
#     ):
#         return np.argmin(self.calculate_distance(X), axis=1)
