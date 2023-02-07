from dataclasses import dataclass
import os
from pathlib import Path

from PIL import Image
import numpy as np


@dataclass
class KMeans:
    file: str
    k: int
    image_height: int = 30

    def __post_init__(self):
        self.output_directory = Path(self.file).with_suffix("")
        self.output_directory.mkdir(exist_ok=True, parents=True)

    def fit(self):
        with Image.open(self.file) as f:
            pixels = np.asarray(f)
        self.pixels = pixels.reshape(
            (pixels.shape[0] * pixels.shape[1], pixels.shape[2])
        ).T

        self.labels, self.centroids = self.get_kmeans()
        self.modes = self.calculate_modes()
        self.ordered_clusters = self.get_ordered_clusters()
        self.proportional_centroids = self.get_proportional_matrix(self.centroids)
        self.proportional_modes = self.get_proportional_matrix(self.modes)

    def output(self):
        self.write_proportional_image(
            self.proportional_centroids, "centroids_palette.png"
        )
        self.write_proportional_image(self.proportional_modes, "modes_palette.png")

    def get_kmeans(self):
        centroids = self.pixels[
            :, np.random.randint(self.pixels.shape[1], size=(1, self.k))[0]
        ]

        # initialize previous centroids to check for change
        last_centroids = np.array([])
        # initialize labels in case algorithm fails
        labels = np.zeros(self.pixels.shape[1]).astype(np.uint8)

        # iterate until no change
        while not np.array_equal(centroids, last_centroids):
            # set increments
            last_centroids = centroids
            # calculate labels,
            # check that all clusters are used,
            # and calculate centroids
            labels = self.calculate_distance_labels(centroids)
            centroids = self.calculate_centroids(labels)
        return labels, centroids

    def calculate_centroids(self, labels: np.ndarray) -> np.ndarray:
        """
        Calculate centroids based on input matrix
        of observations and corresponding cluster
        assignments

        Parameters
        ----------
        labels: np.ndarray
            Vector of cluster assignments whose indices correspond
            to the observations

        Returns
        -------
        centroids: np.ndarray
            Array of new centroids

        """
        centroids = np.array(
            [np.mean(self.pixels.T[labels == i], axis=0) for i in range(self.k)]
        ).T
        return centroids

    def calculate_distance_labels(self, centroids: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance of observations to centroids

        Parameters
        ----------
        centroids: np.ndarray
            Array of centroids

        Returns
        -------
        labels: np.ndarray
            Array of cluster labels
        """
        distance = np.array(
            [
                np.linalg.norm(centroids[:, i] - self.pixels.T, axis=1)
                for i in range(self.k)
            ]
        ).T
        labels = np.argmin(distance, axis=1)
        return labels

    def calculate_mode(self, cluster):
        """
        cluster: np.ndarray
            Matrix with column-based observations
        """
        vals, counts = np.unique(cluster, axis=0, return_counts=True)
        cluster_mode = vals[counts == np.max(counts)][0]
        return cluster_mode

    def calculate_modes(self):
        return np.array(
            [
                self.calculate_mode(self.pixels.T[self.labels == i])
                for i in range(self.k)
            ]
        ).T

    def get_ordered_clusters(self):
        _, counts = np.unique(self.labels, return_counts=True)
        idx_sorted = np.argsort(counts)
        ordered_clusters = np.flip(idx_sorted)
        return ordered_clusters

    def get_proportional_matrix(self, matrix):
        return np.concatenate(
            [
                np.repeat(
                    matrix[:, i].reshape(1, -1),
                    int(round(100 * np.mean(self.labels == i))),
                    axis=0,
                )
                for i in self.ordered_clusters
            ]
        )

    def write_proportional_image(self, proportional_array, filename):
        arr = proportional_array.reshape((1, *proportional_array.shape))
        outfile = os.path.join(self.output_directory, filename)
        with Image.fromarray(
            np.repeat(arr, self.image_height, axis=0).astype(np.uint8)
        ) as f:
            f.save(outfile)
