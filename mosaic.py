from numpy.typing import NDArray
from typing import Optional
import numpy as np

from shape import Shape
from fit import fit
import arrays
import images


def create(n_target_pixels: int,
           tile_height: int,
           target_src: Optional[str | bytes] = None,
           tiles_src: Optional[str | list[bytes]] = None,
           by_pixel: bool = True
           ) -> NDArray:
    """

    """
    target = images.preprocess(src=target_src, n_pixels=n_target_pixels, tile_height=tile_height)

    tile_shape = Shape(tile_height, tile_height, images.N_COLOURS)
    output_shape = Shape(target.size[1], target.size[0], images.N_COLOURS)

    target_array = np.asarray(target, dtype=np.uint8)
    target_array = arrays.unroll(target_array, tile_shape)

    tile_arrays = images.tiles_as_arrays(tile_shape, tiles_src)

    best_comparison = fit(target_array.astype(np.uint8), tile_arrays.astype(np.uint8), by_pixel=by_pixel)
    results_array = tile_arrays[best_comparison]
    return arrays.rollup(results_array, tile_shape, output_shape)
