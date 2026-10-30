def apply_actions(df, anomaly_report, actions,
                  negative_report, negative_actions,
                  missing_id_report, retained_missing_id_rows):
    """
    Applies user-selected actions to the DataFrame.

    Pass 1 — Apply per-column negative value actions
    Pass 2 — Apply column-level anomaly actions
    Pass 3 — Remove missing ID rows except retained ones
    """

    preprocessing_log = {
        "values_replaced_median": 0,
        "values_replaced_zero": 0,
        "rows_deleted_negative": 0,
        "rows_adjusted_winsorize": 0,
        "rows_excluded_anomaly": 0,
        "rows_removed_missing_id": 0
    }

    rows_to_exclude = set()

    # Pass 1 — Per-column negative value actions
    negative_columns = negative_report.get("columns", {})

    for col, action in negative_actions.items():

        if col not in negative_columns:
            continue

        if col not in df.columns:
            continue

        stats = negative_columns[col]
        negative_mask = df[col] < 0
        affected_rows = df[negative_mask].index.tolist()

        if action == "Replace with Median":
            median_val = df[col][df[col] >= 0].median()
            df.loc[negative_mask, col] = median_val
            preprocessing_log["values_replaced_median"] += len(affected_rows)

        elif action == "Replace with 0":
            df.loc[negative_mask, col] = 0
            preprocessing_log["values_replaced_zero"] += len(affected_rows)

        elif action == "Delete Rows":
            rows_to_exclude.update(affected_rows)
            preprocessing_log["rows_deleted_negative"] += len(affected_rows)

    # Collect all negative row indices for Pass 2 reference
    all_negative_indices = set()
    for col_stats in negative_columns.values():
        for idx in col_stats.get("rows", []):
            all_negative_indices.add(int(idx))

    # Pass 2 — Column-level anomaly actions
    for col, action in actions.items():

        if col not in anomaly_report:
            continue

        if col not in df.columns:
            continue

        stats = anomaly_report[col]

        non_negative_rows = [
            idx for idx in stats["rows"]
            if idx in df.index
            and idx not in all_negative_indices
        ]

        if not non_negative_rows:
            continue

        if action == "Retain Records":
            continue

        elif action == "Adjust Values":
            for idx in non_negative_rows:
                if idx not in rows_to_exclude:
                    val = df.loc[idx, col]
                    if val > stats["upper_limit"]:
                        df.loc[idx, col] = stats["upper_limit"]
                    elif val < stats["lower_limit"]:
                        df.loc[idx, col] = stats["lower_limit"]
            preprocessing_log["rows_adjusted_winsorize"] += len(non_negative_rows)

        elif action == "Exclude Records":
            rows_to_exclude.update(non_negative_rows)
            preprocessing_log["rows_excluded_anomaly"] += len(non_negative_rows)

    # Pass 3 — Missing ID rows
    all_missing_id_indices = set(
        int(row_key) for row_key in missing_id_report.keys()
    )
    retained_missing_indices = set(
        int(idx) for idx in retained_missing_id_rows
    )
    missing_to_remove = all_missing_id_indices - retained_missing_indices
    rows_to_exclude.update(missing_to_remove)
    preprocessing_log["rows_removed_missing_id"] += len(missing_to_remove)

    # Apply all exclusions at once
    if rows_to_exclude:
        valid_rows = [r for r in rows_to_exclude if r in df.index]
        df = df.drop(index=valid_rows)
        df = df.reset_index(drop=True)

    return df, preprocessing_log
