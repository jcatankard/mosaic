from urllib.request import urlopen
from shape import Shape
from PIL import Image
from fit import fit
import numpy as np
import unittest
import images
import arrays


class TestApp(unittest.TestCase):
    img = Image.open(urlopen(images.EXAMPLE_URL))

    def test_resize(self):
        tile_height = np.random.randint(2, 33)
        n_pixels = np.random.randint(int(10e3), int(10e7))

        new_img = images.resize(self.img, tile_height=tile_height, n_pixels=n_pixels)

        self.assertEqual(new_img.size[0] % tile_height, 0)
        self.assertEqual(new_img.size[1] % tile_height, 0)

    def test_roll_unroll(self):
        tile_height = np.random.randint(2, 33)
        n_pixels = np.random.randint(int(10e3), int(10e7))

        tile_shape = Shape(tile_height, tile_height, images.N_COLOURS)

        new_img = images.resize(self.img, tile_height=tile_height, n_pixels=n_pixels)
        array = np.asarray(new_img, dtype=np.uint8)

        unrolled_array = arrays.unroll(array, tile_shape)
        rollback_array = arrays.rollup(unrolled_array, tile_shape, Shape(*array.shape))

        np.testing.assert_array_equal(array, rollback_array)

    def test_fit(self):
        """
        Create an array with all values set to 255.
        Fit against a range of greyscale values.
        Fit should match against the darkest greyscale array value, which should be also 255
        """
        rgb_value = 255

        tile_height = np.random.randint(2, 33)
        tile_shape = Shape(tile_height, tile_height, images.N_COLOURS)

        n_vertical_tiles = np.random.randint(10, 100)
        n_horizontal_tiles = np.random.randint(10, 100)
        shape = Shape(n_vertical_tiles * tile_height, n_horizontal_tiles * tile_height, images.N_COLOURS)

        target_array = np.ones(shape, dtype=np.uint8) * np.uint8(rgb_value)
        target_array = arrays.unroll(target_array, tile_shape)

        tile_arrays = images.create_greyscale_arrays(tile_shape)
        best_comparison = fit(target_array.astype(np.uint8), tile_arrays.astype(np.uint8), by_pixel=True)

        # check that all values are the same
        self.assertEqual(np.unique(best_comparison).size, 1)

        # check that value is same as rgb value
        value = tile_arrays[best_comparison].flatten()[0]
        self.assertEqual(value, rgb_value)
