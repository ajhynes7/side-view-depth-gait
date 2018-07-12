import numpy as np
import modules.signals as sig


def test_root_mean_square():

    x = np.array([0, 1, 2])
    assert np.isclose(sig.root_mean_square(x), np.sqrt(5 / 3))

    x = np.array([0, 1, 2, 3])
    assert np.isclose(sig.root_mean_square(x), np.sqrt(14 / 4))


def test_normalize():

    for i in range(10):

        dim = np.random.randint(2, 5)  # Vector dimension
        v = np.random.uniform(0, 10, dim)

        normalized = sig.normalize(v)

        in_range = np.logical_and(np.all(normalized >= 0),
                                  np.all(normalized <= 1))

        assert in_range