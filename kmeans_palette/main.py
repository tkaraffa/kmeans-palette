"""
todo

order clusters/modes by size (do this once for both)

display colors by prevalance - ie, scale outputs to match how big the cluster is

add pyproject.toml kmeans-palette should run main

more cli args

docstrings, argparse usage, docs, etc

output palette to terminal??? colors, codes, etc
"""

import os
import argparse

from PIL import Image
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("-k", type=int, default=5)

    return parser.parse_args()


def get_kmeans(x: np.ndarray, k: int):
    centroids = x[:, np.random.randint(x.shape[1], size=(1, k))[0]]

    # initialize previous centroids to check for change
    last_centroids = np.array([])
    # initialize labels in case algorithm fails
    labels = np.zeros(x.shape[1]).astype(np.uint8)

    # iterate until no change
    while not np.array_equal(centroids, last_centroids):
        # set increments
        last_centroids = centroids
        # calculate labels,
        # check that all clusters are used,
        # and calculate centroids
        labels = calculate_distance_labels(x, centroids, k)
        centroids = calculate_centroids(x, labels, k)
    return labels, centroids


def calculate_centroids(
        x: np.ndarray, labels: np.ndarray, k: int
) -> np.ndarray:
    """
    Calculate centroids based on input matrix
    of observations and corresponding cluster
    assignments

    Parameters
    ----------
    x: np.ndarray
        Matrix of observations
    labels: np.ndarray
        Vector of cluster assignments whose indices correspond
        to the observations
    k: int
        The number of clusters

    Returns
    -------
    centroids: np.ndarray
        Array of new centroids

    """
    centroids = np.array(
        [np.mean(x.T[labels == i], axis=0) for i in range(k)]
    ).T
    return centroids


def calculate_distance_labels(
        x: np.ndarray, centroids: np.ndarray, k: int
) -> np.ndarray:
    """
    Compute Euclidean distance of observations to centroids

    Parameters
    ----------
    x: np.ndarray
        Array of observations
    centroids: np.ndarray
        Array of centroids
    k: int
        The number of clusters

    Returns
    -------
    labels: np.ndarray
        Array of cluster labels
    """

    distance = np.array(
        [
            np.linalg.norm(centroids[:, j] - x.T, axis=1)
            for j in range(k)
        ]
    ).T
    labels = np.argmin(distance, axis=1)
    return labels


def calculate_modes(x, labels, k):
    return np.array([
        calculate_mode(x.T[labels == i]) for i in range(k)
    ])


def calculate_mode(cluster):
    """
    cluster: np.ndarray
        Matrix with column-based observations
    """
    vals, counts = np.unique(cluster, axis=0, return_counts=True)
    cluster_mode = vals[counts == np.max(counts)][0]
    return cluster_mode

def get_ordered_clusters(labels):
    _, counts = np.unique(labels, return_counts=True)
    idx_sorted = np.argsort(counts)
    ordered_clusters = np.flip(idx_sorted)    
    return ordered_clusters

def main():

    args = get_args()

    k = args.k
    file = args.file

    with Image.open(file) as f:
        pixels = np.asarray(f)
    original_dimensions = pixels.shape
    pixels = pixels.reshape((pixels.shape[0] * pixels.shape[1], pixels.shape[2])).T

    labels, centroids = get_kmeans(pixels, k)
    ordered_clusters = get_ordered_clusters(labels)

    # for i in ordered_clusters:
    #     print(centroids[:, i], int(100*np.mean(labels==i)))
    z=np.array([[None]])
    for i in ordered_clusters:
        p = np.repeat(
            centroids[:, i].reshape(-1, 1), round(100*np.mean(labels==i), 0), axis=1
        )
        z = np.append(z, p, axis=0)
        # print(p.shape)

    print(z.shape)
    print(z.reshape( 3, int(z.shape[0]/3)))

    # proportional_centroids = np.array(
    #     [
    #         np.repeat(
    #         centroids[:, i].reshape(-1, 1), int(100*np.mean(labels==i)), axis=1
    #     ) for i in ordered_clusters
    #     ]
    # )
    # print(proportional_centroids)

    centroids = centroids.T.reshape((1, k, original_dimensions[2]))
    with Image.fromarray(centroids.astype(np.uint8)) as f:
        f.save("centroid_palette.png")

    modes = calculate_modes(pixels, labels, k)
    modes = modes.reshape((1, k, original_dimensions[2]))
    with Image.fromarray(modes.astype(np.uint8)) as f:
        f.save("mode_palette.png")


if __name__ == "__main__":
    main()
