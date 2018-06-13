import numpy as np
import modules.math_funcs as mf


def test_normalize_array():

    for i in range(10):

        dim = np.random.randint(2, 5)  # Vector dimension
        v = np.random.uniform(0, 10, dim)

        normalized = mf.normalize_array(v)

        in_range = np.logical_and(np.all(normalized >= 0),
                                  np.all(normalized <= 1))

        assert in_range


def test_norm_ratio():

    for i in range(10):

        a = np.random.randint(1, 10)
        b = np.random.randint(1, 10)

        r = mf.norm_ratio(a, b)

        assert r > 0 and r <= 1

    assert np.isnan(mf.norm_ratio(0, 5))
