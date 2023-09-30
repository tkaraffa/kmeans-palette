from typing import Optional
from dataclasses import dataclass, field
from collections import namedtuple
import os
from pathlib import Path

from PIL import Image
import numpy as np

# namedtuple for holding central tendency configurations
Central = namedtuple(
    "Central",
    ["attribute", "proportional_attribute", "title", "image", "alt"],
)


@dataclass
class KMeans:
    file: str
    k: int
    image_width: int
    image_height: int
    output_directory: str = field(default=os.getcwd())
    centroids_only: bool = field(default=False)
    modes_only: bool = field(default=False)

    @property
    def centroid_attributes(self) -> Optional[Central]:
        if not self.modes_only:
            return Central(
                "centroids",
                "proportional_centroids",
                "Centroids",
                "centroids_palette.png",
                "Proportional Centroids",
            )

    @property
    def mode_attributes(self) -> Optional[Central]:
        if not self.centroids_only:
            return Central(
                "modes",
                "proportional_modes",
                "Modes",
                "modes_palette.png",
                "Proportional Modes",
            )

    @property
    def attributes(self) -> list[Central]:
        if self.centroids_only and self.modes_only:
            raise AttributeError(
                "Choose, at most, one of `centroids_only` and `modes_only`."
            )
        return filter(
            None,
            [self.centroid_attributes, self.mode_attributes],
        )

    @property
    def full_output_directory(self) -> str:
        """
        Creates directory for outputting color palette files
        """
        full_output_directory = os.path.join(
            self.output_directory, Path(self.file).with_suffix("").stem
        )
        Path(full_output_directory).mkdir(exist_ok=True, parents=True)
        return full_output_directory

    def fit_kmeans(self):
        """
        Calculate k-means clusters, and optionally sets proportional
        arrays for centroids and modes
        """
        self.pixels = self.read_image_as_pixels()
        self.labels, self.centroids = self.compute_kmeans()
        self.ordered_clusters = self.get_ordered_clusters()

        if not self.centroids_only:
            self.modes = self.calculate_modes()
            self.proportional_modes = self.get_proportional_matrix(self.modes)
        if not self.modes_only:
            self.proportional_centroids = self.get_proportional_matrix(
                self.centroids
            )

        for attribute in self.attributes:
            self.write_proportional_images(attribute)
        self.write_markdown(
            os.path.join(
                self.full_output_directory,
                Path(os.path.basename(self.file)).with_suffix(".md"),
            )
        )

    def read_image_as_pixels(self):
        """
        Reshapes 3D array of color codes/image width/image height
        into a 2D array of color codes/image width * image height
        """
        with Image.open(self.file) as f:
            pixels = np.asarray(f)
        return pixels.reshape(
            (pixels.shape[0] * pixels.shape[1], pixels.shape[2])
        ).T

    def compute_kmeans(self):
        """
        Compute k-means clusters based on array of pixels
        """
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

    @staticmethod
    def calculate_mode(cluster):
        """
        cluster: np.ndarray
            Matrix with column-based observations
        """
        vals, counts = np.unique(cluster, axis=0, return_counts=True)
        cluster_mode = vals[counts == np.max(counts)][0]
        return cluster_mode

    def calculate_modes(self) -> np.ndarray:
        """
        Calculate modes of clusters
        """
        modes = np.array(
            [
                self.calculate_mode(self.pixels.T[self.labels == i])
                for i in range(self.k)
            ]
        ).T
        return modes

    def get_ordered_clusters(self):
        """
        Obtain index of clusters in descending order based on count
        """
        _, counts = np.unique(self.labels, return_counts=True)
        idx_sorted = np.argsort(counts)
        ordered_clusters = np.flip(idx_sorted)
        return ordered_clusters

    def get_color_codes(self, colors: np.ndarray):
        """
        Obtain formatted RGB and hex color codes of a 2D array
        """
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
        """
        Obtain 2D matrix of RGB color codes with number of triplets
        proportional to representation of that color code in clusters
        """
        proportional_matrix = np.concatenate(
            [
                np.repeat(
                    matrix[:, i].reshape(1, -1),
                    int(round(self.image_width * np.mean(self.labels == i))),
                    axis=0,
                )
                for i in self.ordered_clusters
            ]
        )
        return proportional_matrix

    def write_proportional_images(self, attribute: Central):
        """
        Writes proportional matrices of colors to images for
        relevant attributes
        """
        try:
            outfile = os.path.join(self.full_output_directory, attribute.image)
            self.write_proportional_image(
                getattr(self, attribute.proportional_attribute),
                outfile,
            )
        except AttributeError:
            pass

    def write_proportional_image(self, proportional_array, outfile):
        """
        Reshape array of RGB color codes to 3D matrix
        and write to 2D image
        """
        arr = proportional_array.reshape((1, *proportional_array.shape))
        shaped_arr = np.repeat(arr, self.image_height, axis=0).astype(np.uint8)
        with Image.fromarray(shaped_arr) as f:
            f.save(outfile)

    def write_markdown(self, outfile):
        """
        Write markdown to display color palettes and
        RGB/hex codes of colors for each cluster
        """
        with open(outfile, "w") as f:
            for attribute in self.attributes:
                f.write("\n")
                f.write(
                    self.write_markdown_section(
                        title=attribute.title,
                        image=attribute.image,
                        alt=attribute.alt,
                        colors=getattr(self, attribute.attribute),
                    )
                )

    def write_markdown_section(self, title, image, alt, colors):
        """
        Format text for markdown section to include
        link to image and formatted table of
        cluster, RGB, and hex codes
        """
        output = f"## {title}\n\n"
        img_template = '<img src="{image}" alt="{alt}" height="{height}" width="{width}">\n\n'  # noqa

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
