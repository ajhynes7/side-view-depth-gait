"""Unit tests for intraclass correlation coefficients."""

import numpy as np
import pytest

from analysis.icc import (
    anova_sum_squares,
    anova_mean_squares,
    SumSquares,
    MeanSquares,
    icc,
)


@pytest.fixture
def X_clinical() -> np.ndarray:

    return np.array(
        [
            [59.9, 67.7, 72.2],
            [62.9, 66.5, 67.9],
            [58.9, 50.1, 47.9],
            [46.8, 50.0, 53.9],
            [62.5, 67.8, 62.6],
            [44.8, 42.7, 48.4],
            [57.3, 49.6, 48.0],
            [49.0, 45.2, 57.5],
            [43.5, 41.5, 47.4],
            [39.2, 50.9, 56.3],
        ]
    )


@pytest.mark.parametrize(
    "x, y, ss_true, ms_true, iccs_true",
    [
        (
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            SumSquares(20, 0, 0, 20, 20, 0),
            MeanSquares(5, 0, 0, 2.5, 2.22, 0),
            [1, 1, 1],
        ),
        (
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            SumSquares(20, 2.5, 2.5, 20, 22.5, 0),
            MeanSquares(5, 2.5, 0.5, 2.5, 2.5, 0),
            [0.818, 0.833, 1],
        ),
    ],
)
def test_icc(x, y, ss_true, ms_true, iccs_true):

    X = np.column_stack((x, y))
    n, k = X.shape

    ss = anova_sum_squares(X)
    for ss_val, ss_val_true in zip(ss, ss_true):
        assert ss_val.round(2) == ss_val_true

    ms = anova_mean_squares(ss, n, k)
    for ms_val, ms_val_true in zip(ms, ms_true):
        assert ms_val.round(2) == ms_val_true

    for i in range(3):
        icc_calculated = icc(X, form=i + 1).round(3)
        assert icc_calculated == iccs_true[i]


def test_on_clinical_example(X_clinical: np.ndarray):

    n, k = X_clinical.shape

    ss = anova_sum_squares(X_clinical)
    ms = anova_mean_squares(ss, n, k)

    assert ms.BS.round(2) == 212.61
    assert ms.BM.round(2) == 39.15
    assert ms.WS.round(2) == 25.92

    assert icc(X_clinical, form=1).round(3) == 0.706
    assert icc(X_clinical, form=2).round(3) == 0.708
    assert icc(X_clinical, form=3).round(3) == 0.720
