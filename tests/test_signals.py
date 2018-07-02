import numpy as np
import modules.signals as sig


def test_root_mean_square():

    x = np.array([0, 1, 2])
    assert np.isclose(sig.root_mean_square(x), np.sqrt(5 / 3))

    x = np.array([0, 1, 2, 3])
    assert np.isclose(sig.root_mean_square(x), np.sqrt(14 / 4))

