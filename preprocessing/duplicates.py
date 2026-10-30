def remove_duplicates(df):
    """
    Detects and removes duplicate rows.
    Returns the cleaned DataFrame and a report.
    """

    duplicates_before = df.duplicated().sum()

    df = df.drop_duplicates().reset_index(drop=True)

    duplicates_after = df.duplicated().sum()

    report = {
        "duplicates_before": duplicates_before,
        "duplicates_after": duplicates_after,
        "duplicates_removed": duplicates_before - duplicates_after
    }

    return df, report