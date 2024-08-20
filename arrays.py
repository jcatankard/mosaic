from numpy.typing import NDArray
from shape import Shape
import numpy as np


def unroll(array: NDArray, tile_shape: Shape) -> NDArray:
    """
    :param array: array from image dimensions 3d array (total_height, total_width, N_COLOURS)
    :param tile_shape: shape of array sections (section_height, section_height, N_COLOURS)
    :return: 4d array (n_tiles, tile_section_height, tile_section_height, N_COLOURS)
    """
    count = 0
    n_tiles = array.size // tile_shape.total_size
    new_array = np.empty(shape=(n_tiles, *tile_shape), dtype=np.uint8)
    for i in range(0, array.shape[0], tile_shape.height):
        for j in range(0, array.shape[1], tile_shape.height):
            new_array[count] = array[i: i + tile_shape.height, j: j + tile_shape.height]
            count += 1
    return new_array


def rollup(array: NDArray, tile_shape: Shape, new_array_shape: Shape) -> NDArray:
    """
    :param array: 4d array (n_tiles, tile_section_height, tile_section_height, N_COLOURS)
    :param tile_shape: shape of array that makes the tiles (tile_section_height, tile_section_height, N_COLOURS)
    :param new_array_shape: shape of output array (total_height, total_width, N_COLOURS)
    :return: array of dimensions new_array_shape
    """
    count = 0
    new_array = np.empty(shape=new_array_shape.as_tuple(), dtype=np.uint8)
    for i in range(0, new_array_shape.height, tile_shape.height):
        for j in range(0, new_array_shape.width, tile_shape.height):
            new_array[i: i + tile_shape.height, j: j + tile_shape.height] = array[count]
            count += 1
    return new_array
