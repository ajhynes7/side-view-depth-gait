import pytest
import numpy as np

from util import general as gen


def test_normalize_array():

    for i in range(10):

        dim = np.random.randint(2, 5)  # Vector dimension
        v = np.random.uniform(0, 10, dim)

        normalized = gen.normalize_array(v)

        in_range = np.logical_and(np.all(normalized >= 0),
            np.all(normalized <= 1))

        assert in_range


def test_ratio_func():

    for i in range(10):

        a = np.random.randint(1, 10)
        b = np.random.randint(1, 10)

        assert gen.ratio_func(a, b) >= 1
        