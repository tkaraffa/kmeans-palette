import os
import sys


import pytest


from kmeans import KMeans


class TestKMeans(object):
    def test_init(self):
        kmeans = KMeans(file="test.png")
        assert kmeans.full_output_directory == os.path.join(
            os.getcwd(), "test"
        )
        assert all(
            set(attribute.keys())
            == {"attribute", "proportional_attribute", "title", "image", "alt"}
            for attribute in kmeans.attributes
        )

    def test_fit(self):
        with pytest.raises(AttributeError):
            KMeans(
                file="test.png",
                centroids_only=True,
                modes_only=True,
            ).fit()
