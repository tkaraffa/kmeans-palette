"""
todo

add pyproject.toml kmeans-palette should run main

more cli args

docstrings, argparse usage, docs, etc

output palette to terminal??? colors, codes, etc
"""
import argparse
import os
from pathlib import Path

from PIL import Image
import numpy as np

from kmeans import KMeans


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("-k", type=int, default=5)

    return parser.parse_args()


# def get_kmeans(x: np.ndarray, k: int):
#     centroids = x[:, np.random.randint(x.shape[1], size=(1, k))[0]]

#     # initialize previous centroids to check for change
#     last_centroids = np.array([])
#     # initialize labels in case algorithm fails
#     labels = np.zeros(x.shape[1]).astype(np.uint8)

#     # iterate until no change
#     while not np.array_equal(centroids, last_centroids):
#         # set increments
#         last_centroids = centroids
#         # calculate labels,
#         # check that all clusters are used,
#         # and calculate centroids
#         labels = calculate_distance_labels(x, centroids, k)
#         centroids = calculate_centroids(x, labels, k)
#     return labels, centroids


# def calculate_centroids(x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
#     """
#     Calculate centroids based on input matrix
#     of observations and corresponding cluster
#     assignments

#     Parameters
#     ----------
#     x: np.ndarray
#         Matrix of observations
#     labels: np.ndarray
#         Vector of cluster assignments whose indices correspond
#         to the observations
#     k: int
#         The number of clusters

#     Returns
#     -------
#     centroids: np.ndarray
#         Array of new centroids

#     """
#     centroids = np.array([np.mean(x.T[labels == i], axis=0) for i in range(k)]).T
#     return centroids


# def calculate_distance_labels(
#     x: np.ndarray, centroids: np.ndarray, k: int
# ) -> np.ndarray:
#     """
#     Compute Euclidean distance of observations to centroids

#     Parameters
#     ----------
#     x: np.ndarray
#         Array of observations
#     centroids: np.ndarray
#         Array of centroids
#     k: int
#         The number of clusters

#     Returns
#     -------
#     labels: np.ndarray
#         Array of cluster labels
#     """

#     distance = np.array(
#         [np.linalg.norm(centroids[:, i] - x.T, axis=1) for i in range(k)]
#     ).T
#     labels = np.argmin(distance, axis=1)
#     return labels


# def calculate_mode(cluster):
#     """
#     cluster: np.ndarray
#         Matrix with column-based observations
#     """
#     vals, counts = np.unique(cluster, axis=0, return_counts=True)
#     cluster_mode = vals[counts == np.max(counts)][0]
#     return cluster_mode


# def get_ordered_clusters(labels):
#     _, counts = np.unique(labels, return_counts=True)
#     idx_sorted = np.argsort(counts)
#     ordered_clusters = np.flip(idx_sorted)
#     return ordered_clusters


# def get_proportional_matrix(matrix, ordered_clusters, labels):
#     return np.concatenate(
#         [
#             np.repeat(
#                 matrix[:, i].reshape(1, -1),
#                 int(round(100 * np.mean(labels == i))),
#                 axis=0,
#             )
#             for i in ordered_clusters
#         ]
#     )


def main():
    args = get_args()

    k = args.k
    file = args.file

    kmeans = KMeans(file, k)
    kmeans.fit()
    kmeans.output()


if __name__ == "__main__":
    main()
