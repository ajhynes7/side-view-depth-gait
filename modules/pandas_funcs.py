"""Functions for assisting with the pandas library."""
import pandas as pd

import modules.general as gen


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

    to_drop = [gen.any_in_string(x, strings_to_drop) for x in labels]

    df = df.drop(labels[to_drop], axis=axis)

    return df


if __name__ == "__main__":

    import doctest
    doctest.testmod()
