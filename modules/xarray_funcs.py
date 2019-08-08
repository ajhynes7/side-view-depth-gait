"""Functions related to the xarray library."""

from typing import Callable

import numpy as np
import xarray as xr
from xarray import DataArray


def unique_frames(array_xr: DataArray, func: Callable) -> DataArray:
    """
    Return a DataArray with unique frames as coordinates.

    A function is applied to rows with the same frame to obtain a single row.

    Parameters
    ----------
    array_xr : DataArray
        An array with 'frames' as the first dimension.
    func : function
        Function to convert multiple rows into one row.

    Returns
    -------
    DataArray
        The array with unique frames.

    Examples
    --------
    >>> array_xr = xr.DataArray(
    ...     [[1, 2], [3, 4], [5, 6]],
    ...     coords=([1, 1, 2], range(2)),
    ...     dims=('frames', 'cols'),
    ... )

    >>> array_xr.values
    array([[1, 2],
           [3, 4],
           [5, 6]])

    >>> unique_frames(array_xr, lambda x: np.mean(x, axis=0)).values
    array([[2., 3.],
           [5., 6.]])

    """
    frames_unique = np.unique(array_xr.frames.values)

    def yield_rows():

        for frame in frames_unique:
            rows = array_xr.loc[frame]

            yield rows if rows.ndim == 1 else func(rows)

    array_xr_unique = np.stack([*yield_rows()])

    return xr.DataArray(array_xr_unique, coords=(frames_unique, array_xr.coords['cols']), dims=('frames', 'cols'))
