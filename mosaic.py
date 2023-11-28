from typing import Optional, Union
from numpy.typing import NDArray
from shape import Shape
import numpy as np
import arrays
import images
import fit


def prepare(n_target_pixels: int,
            tile_height: int,
            target_src: Optional[Union[str, bytes]] = None,
            tiles_src: Optional[Union[str, list[bytes]]] = None
            ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], Shape, Shape]:

    target_array = images.preprocess(tile_height=tile_height, n_pixels=n_target_pixels, src=target_src)

    tile_shape = Shape(tile_height, tile_height, images.N_COLOURS)
    output_shape = Shape(*target_array.shape)

    target_array = arrays.unroll(target_array, tile_shape)

    tile_arrays = images.tile_images_as_arrays(tile_shape, tiles_src)
    return target_array, tile_arrays, tile_shape, output_shape


def create(target_array: NDArray[np.uint8],
           tile_arrays: NDArray[np.uint8],
           tile_shape: Shape,
           output_shape: Shape
           ) -> NDArray:
    comparisons = fit.evaluate(rgb_mean(target_array), rgb_mean(tile_arrays))
    best_comparison = fit.best_fit(comparisons)
    results_array = tile_arrays[best_comparison]
    return arrays.rollup(results_array, tile_shape, output_shape)


def rgb_mean(array: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Turns an array of multiple images/tiles and returns the average rgb values for each"""
    return array.sum(axis=1).sum(axis=1) / (array.shape[1] * array.shape[1])
