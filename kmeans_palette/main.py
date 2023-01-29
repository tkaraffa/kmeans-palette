import os

from PIL import Image
import numpy as np
from scipy.stats import mode


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
        labels = get_distance_labels(x, centroids, k)
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

def calculate_modes(x, labels, k):
    modes = np.array(
        [mode(x[labels == i]) for i in range(k)]
    )
    return modes


def get_distance_labels(
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


def main():
    k = 2
    file = os.path.join(os.path.dirname(__file__), "the-wounded-deer.jpg")
    with Image.open(file) as f:
        pixels = np.asarray(f)
    original_dimensions = pixels.shape
    pixels = pixels.reshape((pixels.shape[0] * pixels.shape[1], pixels.shape[2])).T

    labels, centroids = get_kmeans(pixels, k)
    centroids = centroids.T.reshape((k, 1, original_dimensions[2]))

    with Image.fromarray(centroids.astype(np.uint8)) as f:
        f.save("test.png")
    modes = np.array([mode(pixels.T[labels==i], keepdims=False).mode for i in range(k)]).reshape(k, 1, original_dimensions[2])
    with Image.fromarray(modes.astype(np.uint8)) as f:
        f.save("test2.png")


if __name__ == "__main__":
    main()
