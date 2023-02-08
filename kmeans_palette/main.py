"""
todo

tests

add pyproject.toml kmeans-palette should run main

XX more cli args

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
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("-k", type=int, default=KMeansDefaults.K.value)
    parser.add_argument(
        "--image_width", type=int, default=KMeansDefaults.IMAGE_WIDTH.value
    )
    parser.add_argument(
        "--image_height", type=int, default=KMeansDefaults.IMAGE_HEIGHT.value
    )

    return parser.parse_args()


def main():
    args = get_args()

    k = args.k
    file = args.file
    image_width = args.image_width
    image_height = args.image_height

    kmeans = KMeans(file, k, image_width=image_width, image_height=image_height)
    kmeans.fit()
    kmeans.transform()


if __name__ == "__main__":
    main()
