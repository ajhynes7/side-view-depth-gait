"""Tests for images."""

import numpy as np

import analysis.images as im
import hypothesis.strategies as st
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays

pos_floats = st.floats(min_value=0.1, max_value=1e6)

point_3d = arrays('float', (3, ), st.integers(min_value=-1e6, max_value=1e6))


@st.composite
def array_2d(draw):
    """Generate a 2D numpy array."""
    a = draw(st.integers(min_value=2, max_value=50))
    b = draw(st.integers(min_value=2, max_value=50))

    return draw(arrays('float', (a, b), st.floats(allow_nan=False)))


@given(point_3d, pos_floats, pos_floats, pos_floats, pos_floats)
def test_coordinate_conversion(point_real, x_res, y_res, fov_x, fov_y):
    """Test converting between real and projected coordinates."""
    assume(point_real[-1] != 0)

    # Get focal length from image resolution and field of view in degrees
    f_x = im.focal_length(x_res, fov_x)
    f_y = im.focal_length(y_res, fov_y)

    point_proj = im.real_to_image(point_real, x_res, y_res, f_x, f_y)

    point_real_new = im.image_to_real(point_proj, x_res, y_res, f_x, f_y)

    assert np.allclose(point_real, point_real_new, rtol=1e-3)
