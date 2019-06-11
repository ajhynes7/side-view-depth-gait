"""Property tests for gait parameters."""

import numpy as np
from hypothesis import assume, given
from skspatial.tests.property.strategies import consistent_dim, st_point

import modules.gait_parameters as gp


@given(consistent_dim(3 * [st_point]))
def test_pythagorean(points):
    """
    Test that the Pythagorean theorem holds for step length, stride width, and absolute step length.

    step_length ** 2 + stride_width ** 2 == absolute_step_length ** 2

    """
    point_a_i, point_b, point_a_f = points

    assume(not (point_a_i.is_close(point_b) or point_a_i.is_close(point_a_f)))

    dict_spatial = gp.spatial_parameters(point_a_i, point_b, point_a_f)

    hypotenuse = np.sqrt(
        dict_spatial['step_length'] ** 2 + dict_spatial['stride_width'] ** 2
    )

    assert np.isclose(dict_spatial['absolute_step_length'], hypotenuse)
