from urllib.request import urlopen
from numpy.typing import NDArray
from typing import Optional
from shape import Shape
from PIL import Image
import numpy as np
import io


EXAMPLE_URL = "https://live.staticflickr.com/509/31980553422_9b66913640_b.jpg"


class ImageHandler:

    def __init__(self, tile_shape: Shape, n_pixels: int, src: Optional[bytes] = None) -> None:
        """
        :param tile_shape: dimension of tile-image of size (tile_section_height, tile_section_height)
        :param n_pixels: how many pixels to scale the image up or down to
        :param src: bytes to create image from.
        """
        if tile_shape.height != tile_shape.width:
            raise ValueError("Tile shape must be square")
        self.tile_shape = tile_shape

        self.n_pixels = n_pixels
        self.src = src
        self.img = self._process()
        self.array = np.asarray(self.img, dtype=np.uint8)
        self.shape = Shape(*self.array.shape)

    def _process(self) -> Image:
        img = self._open_image()
        img = self._convert_mode(img)
        return self._resize(img)

    def unroll_array(self) -> NDArray[np.uint8]:
        """
        :return: 4d array (n_tiles, tile_section_height, tile_section_height, N_COLOURS)
        """
        count = 0
        n_tiles = self.array.size // self.tile_shape.total_size
        new_array = np.empty(shape=(n_tiles, *self.tile_shape.as_tuple()), dtype=np.uint8)
        for i in range(0, self.array.shape[0], self.tile_shape.height):
            for j in range(0, self.array.shape[1], self.tile_shape.height):
                new_array[count] = self.array[i: i + self.tile_shape.height, j: j + self.tile_shape.height]
                count += 1
        return new_array

    @staticmethod
    def _convert_mode(img: Image) -> Image:
        return img.convert('RGB') if img.mode != 'RGB' else img

    def _resize(self, img: Image) -> Image:
        """
        :param img: image to resize
        :return: resized image
        """
        n_src_pixels = img.size[0] * img.size[1]
        scale = (self.n_pixels / n_src_pixels) ** .5
        size = self.tile_shape.height * (scale * np.array(img.size) / self.tile_shape.height).round(0)
        return img.resize(size=tuple(size.astype(np.int32)))

    def _open_image(self) -> Image:
        if self.src is None:
            img_src = urlopen(EXAMPLE_URL)
        elif isinstance(self.src, str):
            img_src = urlopen(self.src)
        else:
            img_src = io.BytesIO(self.src)
        return Image.open(img_src)
