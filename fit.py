from numba import njit, prange, float32, int64, uint8
from numpy.typing import NDArray
import numpy as np


@njit(float32[::1](uint8[:, :, ::1]), cache=True)
def rgb_average(y: NDArray[np.uint8]) -> NDArray[np.float32]:
    r = y[:, :, 0].mean()
    g = y[:, :, 1].mean()
    b = y[:, :, 2].mean()
    return np.array([r, g, b], dtype=np.float32)


@njit(float32(uint8[:, :, ::1], uint8[:, :, ::1]), cache=True)
def tile_error(y_true: NDArray[np.uint8], y_pred: NDArray[np.uint8]) -> np.float32:
    return np.mean((rgb_average(y_true) - rgb_average(y_pred)) ** 2)


@njit(int64[::1](uint8[:, :, :, ::1], uint8[:, :, :, ::1]), cache=True, parallel=True)
def fit(target: NDArray[np.uint8], tile_options: NDArray[np.uint8]) -> NDArray[np.int64]:
    """
    :param target: arrays representing sections of the target image that we are trying to represent
    :param tile_options: arrays representing images to for the mosaic from
    :return: index of closest matching image for each target image subsection
    """
    n_option_images = tile_options.shape[0]
    n_target_images = target.shape[0]
    err = np.zeros(shape=(n_target_images, n_option_images), dtype=np.float32)

    for i in prange(n_target_images):
        for j in range(n_option_images):
            err[i, j] = tile_error(target[i], tile_options[j])

    return np.argmin(err, axis=1)
