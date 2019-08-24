import numpy as np
import numpy.testing as npt
import xarray as xr

import modules.side_assignment as sa


def test_compute_basis():

    points_head = [
        [0, 60, 210],
        [10, 60, 210],
        [20, 60, 210],
        [30, 60, 210],
        [40, 60, 210],
        [50, 60, 210],
    ]

    points_a = [
        [0, 0, 220],
        [10, 0, 220],
        [20, 0, 220],
        [30, 0, 220],
        [40, 0, 220],
        [50, 0, 220],
    ]

    points_b = [
        [0, 0, 200],
        [10, 0, 200],
        [20, 0, 200],
        [30, 0, 200],
        [40, 0, 200],
        [50, 0, 200],
    ]

    frames = range(6)

    points_stacked = xr.DataArray(
        np.dstack((points_a, points_b, points_head)),
        coords={
            'frames': frames,
            'cols': range(3),
            'layers': ['points_a', 'points_b', 'points_head'],
        },
        dims=('frames', 'cols', 'layers'),
    )

    basis, points_inlier = sa.compute_basis(points_stacked)

    npt.assert_array_equal(basis.origin, [25, 0, 210])
    npt.assert_array_equal(basis.forward, [1, 0, 0])
    npt.assert_array_equal(basis.up, [0, 1, 0])
    npt.assert_array_equal(basis.perp, [0, 0, -1])
