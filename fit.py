from numba import njit, prange, float32, int64, uint8, bool_
from numpy.typing import NDArray
import numpy as np


@njit(float32(uint8[:, :, ::1], uint8[:, :, ::1]), cache=True)
def pixel_error(y_true: NDArray[np.uint8], y_pred: NDArray[np.uint8]) -> np.float32:
    return np.mean((y_true - y_pred) ** 2)


@njit(float32[::1](uint8[:, :, ::1]), cache=True)
def rgb_average(y: NDArray[np.uint8]) -> NDArray[np.float32]:
    r = y[:, :, 0].mean()
    g = y[:, :, 1].mean()
    b = y[:, :, 2].mean()
    return np.array([r, g, b], dtype=np.float32)


@njit(float32(uint8[:, :, ::1], uint8[:, :, ::1]), cache=True)
def tile_error(y_true: NDArray[np.uint8], y_pred: NDArray[np.uint8]) -> np.float32:
    return np.mean((rgb_average(y_true) - rgb_average(y_pred)) ** 2)


@njit(int64[::1](uint8[:, :, :, ::1], uint8[:, :, :, ::1], bool_), cache=True, parallel=True)
def fit(target: NDArray[np.uint8], tile_options: NDArray[np.uint8], by_pixel: bool) -> NDArray[np.int64]:
    """
    :param target: arrays representing sections of the target image that we are trying to represent
    :param tile_options: arrays representing images to for the mosaic from
    :param by_pixel: compares tile by pixel (more detail) otherwise by tile average (better colour)
    :return: index of closest matching image for each target image subsection

    Numba does not support assigning function to a variable.
    Ideally, to avoid repetition and go over if-statement every loop, code would read:
        error_func = pixel_error if by_pixel else tile_error
        for i in prange(n_target_images):
            for j in range(n_option_images):
                err[i, j] = error_func(target[i], tile_options[j])
    """
    n_option_images = tile_options.shape[0]
    n_target_images = target.shape[0]
    err = np.zeros(shape=(n_target_images, n_option_images), dtype=np.float32)

    for i in prange(n_target_images):
        for j in range(n_option_images):
            err[i, j] = pixel_error(target[i], tile_options[j]) if by_pixel else tile_error(target[i], tile_options[j])

    return np.argmin(err, axis=1)
