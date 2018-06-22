

def apply_to_columns(df_1, df_2, func):

    # Columns with numerical data
    numeric_columns_1 = df_1.select_dtypes(include='number').columns
    numeric_columns_2 = df_2.select_dtypes(include='number').columns

    shared_columns = set(numeric_columns_1) & set(numeric_columns_2)

    return {k: func(df_1[k], df_2[k]) for k in shared_columns}
