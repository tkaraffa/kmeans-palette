"""
Create color palettes based on images,
using a k-means clustering algorithm.
"""

"""
todo

tests

add pyproject.toml kmeans-palette should run main

docstrings, argparse usage, docs, etc

output palette to terminal??? colors, codes, etc
"""
import argparse
import os
from pathlib import Path

from PIL import Image
import numpy as np

from kmeans import KMeans
from enums import KMeansDefaults


def get_args():
    parser = argparse.ArgumentParser(prog="kmeans_palette.py", description=__doc__)
    parser.add_argument(
        "file", type=str, help="The image file for which to compute a color palette."
    )
    parser.add_argument(
        "-k",
        "--clusters",
        metavar="K",
        type=int,
        default=KMeansDefaults.K.value,
        action="store",
        help="The number of clusters to use in the k-means algorithm.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        type=str,
        default=os.getcwd(),
        help="The directory in which to output images and color codes.",
    )
    parser.add_argument(
        "-iw",
        "--image_width",
        metavar="N",
        type=int,
        default=KMeansDefaults.IMAGE_WIDTH.value,
        help="The width in pixels of the output image(s).",
    )
    parser.add_argument(
        "-ih",
        "--image_height",
        metavar="N",
        type=int,
        default=KMeansDefaults.IMAGE_HEIGHT.value,
        help="The height in pixels of the output image(s)",
    )

    only_group = parser.add_mutually_exclusive_group()
    only_group.add_argument(
        "-m",
        "--modes_only",
        action="store_true",
        help="Whether or not to only output the modes of clusters.",
    )
    only_group.add_argument(
        "-c",
        "--centroids_only",
        action="store_true",
        help="Whether or not to only output the centroids of clusters.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    kmeans = KMeans(
        file=args.file,
        k=args.clusters,
        image_width=args.image_width,
        image_height=args.image_height,
        output_directory=args.output_directory,
    )
    kmeans.fit(
        centroids_only=args.centroids_only,
        modes_only=args.modes_only,
    )
    kmeans.transform()


if __name__ == "__main__":
    main()
