def dataset_summary(df):

    summary = {

        "rows": df.shape[0],

        "columns": df.shape[1],

        "missing_values": df.isnull().sum().sum(),

        "duplicates": df.duplicated().sum(),

        "numeric_columns":

            len(df.select_dtypes(include="number").columns),

        "categorical_columns":

            len(df.select_dtypes(include="object").columns)

    }

    return summary