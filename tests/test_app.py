from urllib.request import urlopen
from mosaic import Mosaic
from shape import Shape
from PIL import Image
import numpy as np
import unittest
from image_handler import ImageHandler, EXAMPLE_URL
from tile_handler import TileHandler
from scoring import scoring


class TestApp(unittest.TestCase):

    def test_resize(self):
        tile_height = np.random.randint(2, 33)
        n_pixels = np.random.randint(int(10e3), int(10e7))

        ih = ImageHandler(tile_shape=Shape(tile_height, tile_height), n_pixels=n_pixels)

        img = Image.open(urlopen(EXAMPLE_URL))
        new_img = ih._resize(img)

        self.assertEqual(new_img.size[0] % tile_height, 0)
        self.assertEqual(new_img.size[1] % tile_height, 0)

    def test_unroll(self):
        tile_height = np.random.randint(2, 33)
        n_pixels = np.random.randint(int(10e3), int(10e7))

        ih = ImageHandler(tile_shape=Shape(tile_height, tile_height), n_pixels=n_pixels)
        unrolled_array = ih.unroll_array()

        assert ih.array.size == unrolled_array.size
        assert unrolled_array.shape[-3:] == ih.tile_shape.as_tuple()

        np.testing.assert_array_equal(ih.array[:tile_height, :tile_height], unrolled_array[0])
        np.testing.assert_array_equal(ih.array[-tile_height:, -tile_height:], unrolled_array[-1])

    def test_fit(self):
        """
        Create an array with all values set to 255.
        Fit against a range of greyscale values.
        Fit should match against the darkest greyscale array value, which should be also 255
        """
        rgb_value = 255

        tile_height = np.random.randint(2, 33)
        tile_shape = Shape(tile_height, tile_height)

        n_vertical_tiles = np.random.randint(10, 100)
        n_horizontal_tiles = np.random.randint(10, 100)
        shape = Shape(n_vertical_tiles * tile_height, n_horizontal_tiles * tile_height)

        target_array = np.ones(shape.as_tuple(), dtype=np.uint8) * np.uint8(rgb_value)
        target_array = target_array.reshape(-1, *tile_shape.as_tuple())

        tile_arrays = TileHandler(tile_shape, None).array
        comparisons = scoring(Mosaic.rgb_mean(target_array), Mosaic.rgb_mean(tile_arrays))
        best_comparison = Mosaic.best_score(comparisons)

        # check that all values are the same
        self.assertEqual(np.unique(best_comparison).size, 1)

        # check that value is same as rgb value
        value = tile_arrays[best_comparison].flatten()[0]
        self.assertEqual(value, rgb_value)
