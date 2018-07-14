import pytest
import numpy as np
import numpy.testing as npt

import modules.numpy_funcs as nf


def test_divide_no_error():

    assert nf.divide_no_error(10, 5) == 2

    with pytest.raises(Exception):

        nf.divide_no_error(0, 4)
        nf.divide_no_error(4, 0)
        nf.divide_no_error(5, np.nan)


def test_dict_to_array():

    d = {0: 1, 4: 10}

    x = nf.dict_to_array(d)
    y = [1, np.nan, np.nan, np.nan, 10]

    npt.assert_array_equal(x, y)