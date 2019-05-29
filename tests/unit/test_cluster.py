"""Test clustering algorithms."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import modules.cluster as cl


@pytest.fixture
def walking_pass_points():
    """Return toy dataset similar to points in a walking pass."""

    points_cluster = np.array([[0, 0], [1, 0], [0, 1]])

    points_1 = points_cluster
    points_2 = points_cluster + [5, 5]
    points_3 = points_cluster + [0, 10]
    points_4 = points_cluster + [5, 15]

    return np.vstack((points_1, points_2, points_3, points_4))


def test_dbscan(walking_pass_points):
    """Test spatial DBSCAN."""

    labels = cl.dbscan_st(walking_pass_points, eps_spatial=1, min_pts=2)

    labels_expected = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert_array_equal(labels, labels_expected)

    # The default eps is too small, so all points are labelled as noise.
    labels = cl.dbscan_st(walking_pass_points, min_pts=3)
    assert np.all(labels == -1)

    # Change a point to be noise.
    points_noisy = np.append(walking_pass_points, [[5, 10]], axis=0)
    labels_expected.append(-1)

    labels = cl.dbscan_st(points_noisy, eps_spatial=1, min_pts=2)
    assert_array_equal(labels, labels_expected)


def test_dbscan_st(walking_pass_points):
    """Test spatial and temporal DBSCAN."""

    times = np.arange(len(walking_pass_points))
    labels = cl.dbscan_st(
        walking_pass_points,
        times=times,
        eps_spatial=1,
        eps_temporal=5,
        min_pts=2,
    )

    labels_expected = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert_array_equal(labels, labels_expected)

    # One point is marked as noise because of its time.
    times[10] = 1
    labels_expected[10] = -1

    labels = cl.dbscan_st(
        walking_pass_points,
        times=times,
        eps_spatial=1,
        eps_temporal=5,
        min_pts=2,
    )
    assert_array_equal(labels, labels_expected)

    # The points are all labelled as noise if eps_temporal is too low.
    labels = cl.dbscan_st(
        walking_pass_points,
        times=times,
        eps_spatial=1,
        eps_temporal=0.5,
        min_pts=2,
    )
    assert np.all(labels == -1)
