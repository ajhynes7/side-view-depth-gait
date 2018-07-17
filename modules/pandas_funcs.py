"""Functions for assisting with the pandas library."""

from functools import reduce

import pandas as pd

import modules.string_funcs as sf


def swap_columns(df, column_1, column_2):
    """
    Return a copy of a DataFrame with the values of two columns swapped.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column_1, column_2 : str
        Names of the two columns to be swapped.

    Returns
    -------
    df_swapped : DataFrame
        DataFrame with swapped columns

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    >>> swap_columns(df, 'A', 'B')
       A  B  C
    0  4  1  7
    1  5  2  8
    2  6  3  9

    """
    df_swapped = df.copy()

    df_swapped[[column_1, column_2]] = df[[column_2, column_1]]

    return df_swapped


def apply_to_columns(df_1, df_2, func):
    """
    Apply a function on each pair of matching columns from two DataFrames.

    Rows with NaN are removed before applying the function.

    Parameters
    ----------
    df_1, df_2 : DataFrame
        Input DataFrames.
    func : function
        Function that takes two numerical series as inputs.

    Returns
    -------
    dict_ : dict
        Each key is a column label.
        Each value is the output of the given function.

    Examples
    --------
    >>> df_1 = pd.DataFrame({'A': [5, 3], 'B': [2, 10]})
    >>> df_2 = pd.DataFrame({'A': [6, 4], 'B': [3, 2]})

    >>> dict_ = apply_to_columns(df_1, df_2, lambda a, b: a + b)

    >>> dict_['A']
    0    11
    1     7
    Name: A, dtype: int64

    >>> dict_['B']
    0     5
    1    12
    Name: B, dtype: int64

    """
    # Columns with numerical data
    numeric_columns_1 = df_1.select_dtypes(include='number').columns
    numeric_columns_2 = df_2.select_dtypes(include='number').columns

    shared_columns = set(numeric_columns_1) & set(numeric_columns_2)

    dict_ = {}
    for col in shared_columns:

        df_concat = pd.concat([df_1[col], df_2[col]], axis=1).dropna(axis=0)

        dict_[col] = func(df_concat.iloc[:, 0], df_concat.iloc[:, 1])

    return dict_


def lookup_values(df, df_lookup):
    """
    Use values of a DataFrame as the index to a lookup DataFrame.

    The columns of the two tables must match.

    Parameters
    ----------
    df : Dataframe
        Input DataFrame. Values are used as the index to the lookup DataFrame.
    df_lookup : DataFrame
        Lookup DataFrame.

    Returns
    -------
    df_final : DataFrame
        Final DataFrame containing values from the lookup DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'R': [1, 2, 5, 7], 'G': [8, 3, 10, -1]})

    >>> df
       R   G
    0  1   8
    1  2   3
    2  5  10
    3  7  -1

    >>> lookup_data = {'R': [10, 11, 12], 'G': [10, 13, None]}
    >>> df_lookup = pd.DataFrame(lookup_data, index=[1, 5, 7])

    >>> df_lookup
        R     G
    1  10  10.0
    5  11  13.0
    7  12   NaN

    >>> lookup_values(df, df_lookup)
          R   G
    0  10.0 NaN
    1   NaN NaN
    2  11.0 NaN
    3  12.0 NaN

    """
    # Blank DataFrame with same rows and columns
    df_final = pd.DataFrame().reindex_like(df)

    for column in df.columns:

        # Values for the index in the lookup DataFrame
        lookup_index = df[column].values

        looked_up = df_lookup.reindex(lookup_index).loc[:, column].values
        df_final[column] = looked_up

    return df_final


def column_from_lookup(df, df_lookup, *, column='new', lookup_cols=(0, 1)):
    """
    Add a column to a DataFrame and fill it with values from a lookup table.

    For each row in the input DataFrame, the values at two specified columns
    are used as a (row, col) pair to retrieve a new value in the lookup table.

    The retrieved value is inserted into the new column.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    df_lookup : DataFrame
        Dataframe for looking up values.
    column : str, optional
        Name of new column (default 'new').
    lookup_cols : tuple, optional
        The two column fields in the input DataFrame used to look up a value in
        the lookup DataFrame (default (0, 1)).

    Returns
    -------
    df : DataFrame
        Original DataFrame with an added column.

    Examples
    --------
    >>> df = pd.DataFrame({'type': ['x', 'x'], 'make': ['A', 'B']})
    >>> df_lookup = pd.DataFrame({'A': [5, 3], 'B': [2, 10]}, index=['x', 'y'])

    >>> column_from_lookup(df, df_lookup, lookup_cols=('type', 'make'))
      type make  new
    0    x    A    5
    1    x    B    2

    """
    df_copy = df.copy()
    dict_ = {}

    col_1, col_2 = lookup_cols

    for tup in df.itertuples():

        lookup_index = getattr(tup, col_1)
        lookup_col = getattr(tup, col_2)

        value = df_lookup.loc[lookup_index, lookup_col]

        dict_[tup.Index] = value

    df_copy[column] = pd.Series(dict_)

    return df_copy


def drop_any_like(df, strings_to_drop, axis=0):
    """
    Drop labels that contain any of the input strings (case sensitive).

    Rows or columns can be dropped.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    strings_to_drop : iterable
        Sequence of strings to drop from the axis labels.
        A label that contains any of these strings will be dropped.
    axis : int, optional
        {index (0), columns (1)} (default 0).

    Returns
    -------
    DataFrame
        DataFrame with dropped rows or columns.

    Examples
    --------
    >>> df = pd.DataFrame({'Canada': [5, 3], 'UK': [2, 10]}, index=['A', 'B'])

    >>> drop_any_like(df, ['Can'], axis=1)
       UK
    A   2
    B  10

    >>> drop_any_like(df, ['B'], axis=0)
       Canada  UK
    A       5   2

    """
    labels = getattr(df, df._get_axis_name(axis))

    to_drop = [sf.any_in_string(x, strings_to_drop) for x in labels]

    df = df.drop(labels[to_drop], axis=axis)

    return df


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
        for column, func in column_funcs.items():
            yield groupby_object[column].apply(func).to_frame()

    return merge_multiple(yield_groups(), left_index=True, right_index=True)


def split_column(df, *, column=0, delim=' ', new_columns=None, drop=True):
    """
    Split a column into multiple columns by splitting strings at a delimiter.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column : {int, str}, optional
        Name of column to split (default 0).
    delim : str, optional
        Delimiter used to split column (default ' ').
    new_columns : iterable, optional
        Iterable of new column names (default None).
    drop : bool, optional
        If true, drop the original column that was split (default True).

    Returns
    -------
    df_final : DataFrame
        DataFrame including new columns from the split.

    Examples
    --------
    >>> df = pd.DataFrame(['1_2_3', '4_5_6'])

    >>> split_column(df, column=0, delim='_', drop=False)
           0  0  1  2
    0  1_2_3  1  2  3
    1  4_5_6  4  5  6

    >>> split_column(df, column=0, delim='_')
       0  1  2
    0  1  2  3
    1  4  5  6

    >>> split_column(df, column=0, delim='_', new_columns=['a', 'b', 'c'])
       a  b  c
    0  1  2  3
    1  4  5  6

    """
    df_split = df[column].str.split(delim, expand=True)

    if new_columns is not None:
        df_split.columns = new_columns

    if drop:
        df = df.drop(column, axis=1)

    df_final = pd.concat([df, df_split], axis=1)

    return df_final


def column_to_suffixes(df, *, column=0, merge_column=1,
                       column_vals=['A', 'B']):
    """
    Convert the two values of one column into suffixes of new column names.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column : {int, str}, optional
        Name of column name containing two values (default 0).
    merge_column : {int, str}, optional
        Name of column to merge on (default 1).
    column_vals : list, optional
        The two values of the column (default ['A', 'B']).

    Returns
    -------
    df_suffix : DataFrame
        Final DataFrame with suffixes on new columns.

    Examples
    --------
    >>> d = {'group': ['A', 'A', 'B'], 'value': [1, 2, 3], 'number': [0, 1, 0]}
    >>> df = pd.DataFrame(d)

    >>> a, b, c = 'group' , 'number', ['A', 'B']

    >>> column_to_suffixes(df, column=a, merge_column=b, column_vals=c)
       number  value_A  value_B
    0       0        1      3.0
    1       1        2      NaN

    """
    df_l = df[df[column] == column_vals[0]]
    df_r = df[df[column] == column_vals[1]]

    suffixes = ['_' + x for x in column_vals]

    df_suffix = pd.merge(df_l, df_r, left_on=merge_column,
                         right_on=merge_column, suffixes=suffixes, how='outer')

    df_suffix = drop_any_like(df_suffix, [column], axis=1)
    df_suffix = df_suffix.sort_index(axis=1)

    return df_suffix
