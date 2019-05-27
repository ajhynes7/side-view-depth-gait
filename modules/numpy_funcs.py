"""Functions related to NumPy arrays or operations."""

import numpy as np


def interweave_rows(array_a, array_b):

    n_points, n_dim = array_a.shape

    array_c = np.empty((2 * n_points, n_dim), dtype=array_a.dtype)

    array_c[::2] = array_a
    array_c[1::2] = array_b

    return array_c
