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
        return np.array([ImageHandler(self.tile_shape, n_pixels, f).array for f in self.src])

    def _create_greyscale_arrays(self) -> NDArray[np.uint8]:
        values = np.linspace(0, 255, N_DUMMY_IMAGES, dtype=np.uint8)
        arrays = np.ones(shape=(N_DUMMY_IMAGES, *self.tile_shape.as_tuple()), dtype=np.uint8)
        return np.array([arrays[i] * values[i] for i in range(N_DUMMY_IMAGES)], dtype=np.uint8)
