import numpy as np
import pytest

import modules.linear_algebra as lin


@pytest.mark.parametrize(
    "points, centroid, direction",
    [
        (
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
            np.array([1, 0, 0]),
            np.array([1, 0, 0]),
        ),
        (np.array([[0, 0], [4, 0]]), np.array([2, 0]), np.array([1, 0])),
        (np.array([[0, 0], [0, -10]]), np.array([0, -5]), np.array([0, -1])),
    ],
)
def test_best_fit_line_examples(points, centroid, direction):
    """Test specific examples of best fit line."""
    centroid_calc, direction_calc = lin.best_fit_line(points)

    assert np.allclose(centroid, np.round(centroid_calc, 2))
    assert np.allclose(direction, direction_calc)
