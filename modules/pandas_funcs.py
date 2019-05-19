"""Functions for assisting with the pandas library."""

from functools import reduce

import numpy as np
import pandas as pd

import modules.numpy_funcs as nf


def points_to_dataframe(list_points, columns, index):

    n_rows, n_arrays = len(index), len(list_points)
    array_container = np.full((n_rows, n_arrays), None)

    for i, points in enumerate(list_points):
        for row in range(n_rows):
            array_container[row, i] = points[row]

    return pd.DataFrame(array_container, columns=columns, index=index)


def series_of_rows(array, *, index=None):
    """
    Place the rows of a numpy array into a pandas Series.

    Parameters
    ----------
    array : ndarray
        (n, d) array of n rows with length d.
    index : array_like, optional
        Index of the output series (default None).

    Returns
    -------
    series : Series
        Series with n elements.
        Each element in the series is an array of shape (d, ).

    Examples
    --------
    >>> import numpy as np
    >>> array = np.array([[1, 2], [2, 3], [5, 0], [10, 2]])

    >>> series_of_rows(array)
    0     [1, 2]
    1     [2, 3]
    2     [5, 0]
    3    [10, 2]
    dtype: object

    >>> series_of_rows(array, index=[1, 3, 5, 7])
    1     [1, 2]
    3     [2, 3]
    5     [5, 0]
    7    [10, 2]
    dtype: object

    """
    if index is None:
        index, _ = zip(*enumerate(array))

    series = pd.Series(index=index)
    series = series.apply(lambda x: [])

    for idx, vector in zip(series.index, array):

        series[idx] = vector

    return series


def merge_multiple(dataframes, **kwargs):
    """
    Merge multiple DataFrames together.

    Parameters
    ----------
    dataframes : iterable
        Each element is a DataFrame.

    Returns
    -------
    DataFrame
        Result of merging all DataFrames.

    Examples
    --------
    >>> df_1, df_2 = pd.DataFrame([1, 2, 3]), pd.DataFrame([4, 5, 6])
    >>> df_3, df_4 = pd.DataFrame([7, 8, 9]), pd.DataFrame([10, 11, 12])

    >>> dataframes = [df_1, df_2, df_3, df_4]

    >>> merge_multiple(dataframes, left_index=True, right_index=True)
       0_x  0_y  0_x  0_y
    0    1    4    7   10
    1    2    5    8   11
    2    3    6    9   12

    """
    return reduce(lambda a, b: pd.merge(a, b, **kwargs), dataframes)


def apply_to_grouped(df, groupby_column, column_funcs):
    """
    Apply functions to columns of a groupby object and combine the results.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    groupby_column : str
        Name of column to group.
    column_funcs : dict
        Each key is a column name.
        Each value is a function to apply to the column.

    Returns
    -------
    DataFrame
        Result of applying functions to each grouped column.
        Index is the groupby column.
        Columns are those specified as keys in the dictionary.

    Examples
    --------
    >>> d = {'letter': ['K', 'K', 'J'], 'number': [1, 2, 5], 'age': [1, 2, 3]}
    >>> df = pd.DataFrame(d)

    >>> apply_to_grouped(df, 'letter', {'number': min}).reset_index()
      letter  number
    0      J       5
    1      K       1

    """
    groupby_object = df.groupby(groupby_column)

    def yield_groups():
        """Inner function to yield results of applying func to the column."""
        for column, func in column_funcs.items():
            yield groupby_object[column].apply(func).to_frame()

    return merge_multiple(yield_groups(), left_index=True, right_index=True)


def make_index_consecutive(df):
    """
    Make the values of a DataFrame index all consecutive.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.

    Returns
    -------
    df_consec : DataFrame
        DataFrame with a consecutive index.

    Examples
    --------
    >>> df = pd.DataFrame({'Col': [5, 6, 7]}, index=[1, 2, 4])
    >>> make_index_consecutive(df)
       Col
    1    5
    2    6
    3  NaN
    4    7

    """
    index_vals = df.index.values

    index_consec, _ = nf.make_consecutive(index_vals)

    df_consec = pd.DataFrame(index=index_consec, columns=df.columns)
    df_consec.update(df)

    return df_consec
