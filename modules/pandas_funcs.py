import pandas as pd


def apply_to_columns(df_1, df_2, func):

    # Columns with numerical data
    numeric_columns_1 = df_1.select_dtypes(include='number').columns
    numeric_columns_2 = df_2.select_dtypes(include='number').columns

    shared_columns = set(numeric_columns_1) & set(numeric_columns_2)

    return {k: func(df_1[k], df_2[k]) for k in shared_columns}


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

    >>> lookup_data = {'R': [10, 11, 12], 'G': [10, 13, np.nan]}
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


def column_from_lookup(df, df_lookup, column='new', lookup_cols=(0, 1)):
    """
    Add a new column to a dataframe and populate it with values from a
    lookup table.

    For each row in the input dataframe, the values at two specified columns
    are used to retrieve a new value in the lookup table.
    This value is inserted into the new column.

    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    df_lookup : DataFrame
        Dataframe for looking up values.
    column : str, optional
        Name of new column (default 'new').
    lookup_cols : tuple, optional
        The two column fields in input dataframe used to look up a value in
        the lookup dataframe (default (0, 1)).

    Returns
    -------
    df : DataFrame
        Original dataframe with an added column.

    """
    dict_ = {}

    col_1, col_2 = lookup_cols

    for tup in df.itertuples():

        lookup_index = getattr(tup, col_1)
        lookup_col = getattr(tup, col_2)

        value = df_lookup.loc[lookup_index, lookup_col]

        dict_[tup.Index] = value

    df[column] = pd.Series(dict_)

    return df
