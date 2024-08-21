from image_handler import ImageHandler
from tile_handler import TileHandler
from numpy.typing import NDArray
from scoring import scoring
from typing import Optional
from shape import Shape
import numpy as np


class Mosaic:

    def __init__(
            self,
            n_target_pixels: int,
            tile_height: int,
            target_src: Optional[bytes] = None,
            tiles_src: Optional[list[bytes]] = None,
            noise_level: float = 0.
    ) -> None:
        """
        :param n_target_pixels:
        :param tile_height:
        :param target_src:
        :param tiles_src:
        :param noise_level: value between 0 and 100
        """
        self.n_target_pixels = n_target_pixels
        self.target_src = target_src
        self.tiles_src = tiles_src
        self.noise_level = noise_level / 100
        self.tile_shape = Shape(tile_height, tile_height)

        self.target_image: Optional[ImageHandler] = None
        self.tile_images: Optional[TileHandler] = None

    def prepare(self) -> None:
        self.target_image = ImageHandler(self.tile_shape, self.n_target_pixels, self.target_src)
        self.tile_images = TileHandler(self.tile_shape, self.tiles_src)

    def create(self) -> NDArray:
        target_rgb = self.rgb_mean(self.target_image.unroll_array())
        tiles_rgb = self.rgb_mean(self.tile_images.array)

        scores = scoring(target_rgb, tiles_rgb)
        scores *= np.random.normal(loc=1., scale=self.noise_level, size=scores.shape)
        best_scores = self.best_score(scores)

        return self.tile_images.filter_and_rollup(best_scores, self.tile_shape, self.target_image.shape)

    @staticmethod
    def best_score(scores: NDArray) -> NDArray:
        return np.argmin(scores, axis=1)

    @staticmethod
    def rgb_mean(array: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Takes an array of multiple images/tiles and returns the average rgb values for each"""
        return array.sum(axis=1).sum(axis=1) / (array.shape[1] * array.shape[1])
