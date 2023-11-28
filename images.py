from urllib.request import urlopen
from typing import Optional, Union
from numpy.typing import NDArray
from shape import Shape
from os import listdir
from PIL import Image
import numpy as np
import io

EXAMPLE_URL = 'https://live.staticflickr.com/509/31980553422_9b66913640_b.jpg'
N_DUMMY_IMAGES = 25
N_COLOURS = 3


def create_greyscale_arrays(tile_shape: Shape) -> NDArray[np.uint8]:
    values = np.linspace(0, 255, N_DUMMY_IMAGES, dtype=np.uint8)
    arrays = np.ones(shape=(N_DUMMY_IMAGES, *tile_shape), dtype=np.uint8)
    return np.array([arrays[i] * values[i] for i in range(N_DUMMY_IMAGES)], dtype=np.uint8)


def tile_images_as_arrays(tile_shape: Shape, src: Optional[Union[str, list[bytes]]] = None) -> NDArray[np.uint8]:
    """
    :param tile_shape: shape of individual tiles
    :param src: is list of bytes or a string to a folder or None
    :returns: array of tiles shape(n_tiles, *tile_shape)
    """
    if src is None:
        return create_greyscale_arrays(tile_shape)

    if isinstance(src, str):
        folder = listdir(src)
        files = [src + '/' + p for p in folder]
    else:
        files = src

    n_pixels = tile_shape.height * tile_shape.height
    return np.array([preprocess(tile_height=tile_shape.height, n_pixels=n_pixels, src=f) for f in files])


def preprocess(tile_height: int, n_pixels: int, src: Optional[Union[str, bytes]] = None) -> NDArray[np.uint8]:
    img = open_image(src)
    img = convert_mode(img)
    img = resize(img, tile_height, n_pixels)
    return np.asarray(img, dtype=np.uint8)


def convert_mode(img: Image) -> Image:
    return img.convert('RGB') if img.mode != 'RGB' else img


def resize(img: Image, tile_height: int, n_pixels: int) -> Image:
    """
    :param img: image to resize
    :param tile_height: dimension of tile-image of size (tile_section_height, tile_section_height)
    :param n_pixels: how many pixels to scale the image up or down to
    :return: resized image
    """
    n_src_pixels = img.size[0] * img.size[1]
    scale = (n_pixels / n_src_pixels) ** .5
    size = tile_height * (scale * np.array(img.size) / tile_height).round(0)
    return img.resize(size=tuple(size.astype(np.int32)))


def open_image(src: Optional[Union[str, bytes]] = None) -> Image:
    if (src is None) | isinstance(src, str):
        img_src = urlopen(EXAMPLE_URL)
    elif isinstance(src, bytes):
        img_src = io.BytesIO(src)
    else:
        img_src = src
    return Image.open(img_src)
