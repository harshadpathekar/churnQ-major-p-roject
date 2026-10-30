import pandas as pd


def handle_missing_values(df):
    """
    Automatically handles missing values in the dataset.

    Numeric Columns:
        Fill missing values with Median

    Categorical Columns:
        Fill missing values with Mode
    """

    missing_before = df.isnull().sum().sum()

    numeric_columns = df.select_dtypes(include=["number"]).columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in numeric_columns:
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)

    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            mode_series = df[column].mode()
            if not mode_series.empty:
                df[column] = df[column].fillna(mode_series[0])
            else:
                df[column] = df[column].fillna("Unknown")

    missing_after = df.isnull().sum().sum()

    report = {
        "missing_before": missing_before,
        "missing_after": missing_after,
        "handled_missing": missing_before - missing_after
    }

    return df, report