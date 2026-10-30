import pandas as pd
from config import DOMAIN_RULES, NEGATIVE_VALUE_CELL_THRESHOLD, NEGATIVE_VALUE_COLUMN_THRESHOLD


def detect_outliers(df):
    """
    Detects anomalies using domain rules and 10x median threshold.
    Only flags genuinely impossible values — not legitimate business data.
    Does NOT include negative values — those are handled separately.

    10x median logic:
    A value 10 times the typical column value is almost certainly
    a data entry mistake (e.g. 9770 instead of 977).
    This avoids flagging premium customers with legitimately high charges.
    """

    report = {}
    id_column = _detect_id_column(df)
    selected_cols = _select_numeric_cols(df)

    for col in selected_cols:

        anomaly_mask = pd.Series(False, index=df.index)
        reasons = pd.Series("", index=df.index)

        # Check 1 — Domain boundary violations
        col_lower = col.lower()
        if "age" in col_lower and "age" in DOMAIN_RULES:
            max_val = DOMAIN_RULES["age"]
            age_mask = df[col] > max_val
            anomaly_mask |= age_mask
            reasons[age_mask] = f"Age exceeds {max_val} years"

        # Check 2 — 10x median threshold for extreme data entry errors
        col_median = df[col][df[col] >= 0].median()

        if pd.notna(col_median) and col_median > 0:
            upper_bound = col_median * 10
            extreme_mask = df[col] > upper_bound
            anomaly_mask |= extreme_mask
            reasons[extreme_mask] = (
                f"Value exceeds 10x column median "
                f"({round(col_median, 2)})"
            )

        anomaly_rows = df[anomaly_mask]
        count = int(anomaly_mask.sum())

        if count == 0:
            continue

        percentage = round((count / len(df)) * 100, 2)
        rows = list(anomaly_rows.index)

        if id_column:
            customer_ids = list(anomaly_rows[id_column].astype(str))
        else:
            customer_ids = [str(r) for r in rows]

        records = []
        for idx, row in anomaly_rows.iterrows():
            records.append({
                "customer_id": str(row[id_column]) if id_column else str(idx),
                "value": round(float(row[col]), 2),
                "reason": reasons[idx]
            })

        report[col] = {
            "count": count,
            "percentage": percentage,
            "rows": rows,
            "customer_ids": customer_ids,
            "lower_limit": 0,
            "upper_limit": round(float(col_median * 10), 2),
            "has_negative": False,
            "records": records
        }

    return report


def detect_negative_values(df):
    """
    Detects negative values per column.
    Returns per-column summary with threshold warnings.
    Always returns full structure even when no negatives found.
    """

    id_column = _detect_id_column(df)
    selected_cols = _select_numeric_cols(df)

    # Dataset-level threshold check
    total_cells = len(df) * len(selected_cols) if selected_cols else 1
    total_negatives = sum(
        int((df[col] < 0).sum())
        for col in selected_cols
    )

    dataset_threshold_exceeded = False
    if total_cells > 0:
        dataset_negative_pct = total_negatives / total_cells
        if dataset_negative_pct > NEGATIVE_VALUE_CELL_THRESHOLD:
            dataset_threshold_exceeded = True

    negative_columns = {}

    for col in selected_cols:

        negative_mask = df[col] < 0
        count = int(negative_mask.sum())

        if count == 0:
            continue

        percentage = round((count / len(df)) * 100, 2)
        column_threshold_exceeded = (
            (count / len(df)) > NEGATIVE_VALUE_COLUMN_THRESHOLD
        )

        negative_rows = df[negative_mask]
        rows = list(negative_rows.index)

        if id_column:
            customer_ids = list(negative_rows[id_column].astype(str))
        else:
            customer_ids = [str(r) for r in rows]

        # Median of non-negative values only
        non_neg_median = df[col][df[col] >= 0].median()
        median_value = round(float(non_neg_median), 2) if not pd.isna(non_neg_median) else 0.0

        records = []
        for idx, row in negative_rows.iterrows():
            records.append({
                "customer_id": str(row[id_column]) if id_column else str(idx),
                "value": round(float(row[col]), 2),
                "row_index": int(idx)
            })

        negative_columns[col] = {
            "count": count,
            "percentage": percentage,
            "rows": rows,
            "customer_ids": customer_ids,
            "records": records,
            "median_value": median_value,
            "column_threshold_exceeded": column_threshold_exceeded
        }

    return {
        "dataset_threshold_exceeded": dataset_threshold_exceeded,
        "total_negatives": total_negatives,
        "total_cells": total_cells,
        "columns": negative_columns
    }


def detect_missing_id_rows(df):
    """
    Detects rows where the customer ID column has missing values.
    Assigns placeholder IDs like MissingID001, MissingID002, etc.
    """

    id_column = _detect_id_column(df)

    if id_column is None:
        return {}

    missing_mask = df[id_column].isnull()

    if not missing_mask.any():
        return {}

    missing_rows = df[missing_mask]
    missing_report = {}
    counter = 1

    for idx, row in missing_rows.iterrows():
        placeholder_id = f"MissingID{counter:03d}"
        counter += 1

        row_data = {}
        for col in df.columns:
            if col != id_column:
                val = row[col]
                row_data[col] = (
                    round(float(val), 2)
                    if isinstance(val, float)
                    else str(val)
                    if not isinstance(val, (int, bool))
                    else val
                )

        missing_report[str(idx)] = {
            "row_index": idx,
            "placeholder_id": placeholder_id,
            "row_data": row_data
        }

    return missing_report


def _detect_id_column(df):
    for col in df.columns:
        if df[col].nunique() == len(df):
            return col
    return None


def _select_numeric_cols(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    return [
        col for col in numeric_cols
        if df[col].nunique() > 10
        and df[col].nunique() != len(df)
    ]