from dataclasses import dataclass, field
import os
from pathlib import Path

from PIL import Image
import numpy as np

from enums import KMeansDefaults


@dataclass
class KMeans:
    file: str
    k: int = field(default=KMeansDefaults.K.value)
    image_width: int = field(default=KMeansDefaults.IMAGE_WIDTH.value)
    image_height: int = field(default=KMeansDefaults.IMAGE_HEIGHT.value)
    output_directory: str = field(default=os.getcwd())

    def fit(self, centroids_only=False, modes_only=False):
        if centroids_only and modes_only:
            raise AttributeError(
                "Choose, at most, one of `centroids_only` and `modes_only`."
            )
        with Image.open(self.file) as f:
            pixels = np.asarray(f)
        self.pixels = pixels.reshape(
            (pixels.shape[0] * pixels.shape[1], pixels.shape[2])
        ).T

        self.labels, self.centroids = self.get_kmeans()
        self.ordered_clusters = self.get_ordered_clusters()

        if not self.centroids_only:
            self.modes = self.calculate_modes()
            self.proportional_modes = self.get_proportional_matrix(self.modes)
        if not self.modes_only:
            self.proportional_centroids = self.get_proportional_matrix(
                self.centroids
            )

    def transform(self):
        output_directory = os.path.join(
            self.output_directory, Path(self.file).with_suffix("").stem
        )
        Path(output_directory).mkdir(exist_ok=True, parents=True)
        self.write_proportional_images(output_directory)
        self.write_markdown(
            os.path.join(output_directory, Path(self.file).with_suffix(".md"))
        )

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
        return labels, centroids.astype(np.uint8)

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
            [
                np.mean(self.pixels.T[labels == i], axis=0)
                for i in range(self.k)
            ]
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

    def get_color_codes(self, colors: np.ndarray):
        color_str = "|({red},{green},{blue})|{hex}|"
        hex_str = "#%02x%02x%02x"

        color_codes = [
            color_str.format(
                red=triplet[0],
                green=triplet[1],
                blue=triplet[2],
                hex=hex_str.upper() % tuple(triplet),
            )
            for triplet in colors.T[self.ordered_clusters]
        ]
        return color_codes

    def get_proportional_matrix(self, matrix):
        return np.concatenate(
            [
                np.repeat(
                    matrix[:, i].reshape(1, -1),
                    int(round(self.image_width * np.mean(self.labels == i))),
                    axis=0,
                )
                for i in self.ordered_clusters
            ]
        )

    def write_proportional_images(self, output_directory):
        if hasattr(self, "proportional_centroids"):
            self.write_proportional_image(
                self.proportional_centroids,
                os.path.join(output_directory, "centroids_palette.png"),
            )
        if hasattr(self, "proportional_modes"):
            self.write_proportional_image(
                self.proportional_modes,
                os.path.join(output_directory, "modes_palette.png"),
            )

    def write_proportional_image(self, proportional_array, outfile):
        arr = proportional_array.reshape((1, *proportional_array.shape))
        shaped_arr = np.repeat(arr, self.image_height, axis=0).astype(np.uint8)
        with Image.fromarray(shaped_arr) as f:
            f.save(outfile)

    def write_markdown(self, outfile):
        with open(outfile, "w") as f:
            if hasattr(self, "proportional_centroids"):
                f.write(
                    self.write_markdown_section(
                        "Centroids",
                        "centroids_palette.png",
                        "Centroids Palette",
                        self.centroids,
                    )
                )
                f.write("\n")
            if hasattr(self, "proportional_modes"):
                f.write(
                    self.write_markdown_section(
                        "Modes",
                        "modes_palette.png",
                        "Modes Palette",
                        self.modes,
                    )
                )

    def write_markdown_section(self, title, image, alt, colors):
        output = ""
        img_template = '<img src="{image}" alt="{alt}" height="{height}" width="{width}">\n\n'
        output += f"## {title}\n\n"
        output += img_template.format(
            image=image,
            alt=alt,
            height=self.image_height,
            width=self.image_width,
        )
        output += "|Cluster|RGB|Hex|\n"
        output += "|:---:|:---:|:---:|\n"
        for i, line in enumerate(self.get_color_codes(colors)):
            output += f"|{i+1} {line}\n"
        return output
