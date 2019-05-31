"""Unit tests for pose estimation from multiple joint proposals."""

import numpy as np
import pytest

import modules.pose_estimation as pe
from scripts.main.select_proposals import cost_func


@pytest.fixture
def sample_population():

    population = np.array(
        [
            [20, 80, 250],
            [22, 90, 300],
            [20, 20, 244],
            [30, 30, 250],
            [35, 15, 210],
            [30, 10, 210],
        ]
    )

    labels = np.array([0, 0, 1, 1, 2, 2])

    label_adj_list = {0: {1: 60}, 1: {2: 20}, 2: {}}

    return population, labels, label_adj_list


def test_pop_shortest_paths(sample_population):

    population, labels, label_adj_list = sample_population

    prev, dist = pe.pop_shortest_paths(
        population, labels, label_adj_list, cost_func
    )

    assert prev == {0: np.nan, 1: np.nan, 2: 0, 3: 0, 4: 2, 5: 2}
