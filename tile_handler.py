from image_handler import ImageHandler
from numpy.typing import NDArray
from typing import Optional
from shape import Shape
import numpy as np


N_DUMMY_IMAGES = 25


class TileHandler:

    def __init__(self, tile_shape: Shape, src: Optional[list[bytes]] = None) -> None:
        self.src = src
        self.tile_shape = tile_shape
        self.array = self._create_array()

    def _create_array(self) -> NDArray[np.uint8]:
        """
        :returns: array of tiles shape(n_tiles, *tile_shape)
        """
        if self.src is None:
            return self._create_greyscale_arrays()
        n_pixels = self.tile_shape.height * self.tile_shape.height
        tiles = np.array([ImageHandler(self.tile_shape, n_pixels, f).array for f in self.src])
        return np.unique(tiles, axis=0)

    def _create_greyscale_arrays(self) -> NDArray[np.uint8]:
        values = np.linspace(0, 255, N_DUMMY_IMAGES, dtype=np.uint8)
        arrays = np.ones(shape=(N_DUMMY_IMAGES, *self.tile_shape.as_tuple()), dtype=np.uint8)
        return np.array([arrays[i] * values[i] for i in range(N_DUMMY_IMAGES)], dtype=np.uint8)

    def filter_and_rollup(self, index: NDArray, tile_shape: Shape, new_array_shape: Shape) -> NDArray:
        """
        :param index: identifier for which image to use for each tile
        :param tile_shape: shape of array that makes the tiles (tile_section_height, tile_section_height, N_COLOURS)
        :param new_array_shape: shape of output array (total_height, total_width, N_COLOURS)
        :return: array of dimensions new_array_shape
        """
        array = self.array[index]
        count = 0
        new_array = np.empty(shape=new_array_shape.as_tuple(), dtype=np.uint8)
        for i in range(0, new_array_shape.height, tile_shape.height):
            for j in range(0, new_array_shape.width, tile_shape.height):
                new_array[i: i + tile_shape.height, j: j + tile_shape.height] = array[count]
                count += 1
        return new_array
