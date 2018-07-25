import numpy as np

import modules.point_processing as pp


def test_consecutive_dist():

    lengths = [*pp.consecutive_dist(points)]

    assert np.array_equal(np.round(lengths, 4), [2.2361, 9.0554, 8.1854])


def test_closest_point():

    target = [2, 3, 4]
    close_point, close_index = pp.closest_point(np.array(points), target)

    assert close_index == 3


points = [[1, 2, 3], [2, 2, 5], [-1, 10, 2], [2, 3, 5]]