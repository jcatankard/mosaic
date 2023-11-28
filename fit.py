from numba import njit, float64, int64, prange
from numpy.typing import NDArray
import numpy as np


@njit(float64[:, ::1](float64[:, ::1], float64[:, ::1]), cache=True, parallel=True)
def evaluate(target: NDArray[np.float64], tile_options: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    :param target: arrays representing sections of the target image that we are trying to represent
    :param tile_options: arrays representing images to for the mosaic from
    :return: the Euclidean error between the target tile and each tile option
    """
    n_option_images = tile_options.shape[0]
    n_target_images = target.shape[0]
    err = np.zeros(shape=(n_target_images, n_option_images), dtype=np.float64)

    for i in prange(n_target_images):
        for j in range(n_option_images):
            err[i, j] = np.mean((target[i] - tile_options[j]) ** 2)
    return err


@njit(int64[::1](float64[:, ::1]), cache=True)
def best_fit(comparison: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.argmin(comparison, axis=1)
