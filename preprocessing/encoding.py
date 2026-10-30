import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder


def encode_features(df, target_column="Churn", id_column=None,
                    filename=None, processed_folder=None):
    """
    Encodes categorical features for ML model training.

    Strategy:
    - Binary columns (2 unique values): Label Encoding -> 0/1
    - Multi-category columns (3+ unique values): One Hot Encoding
    - Numeric columns: no change
    - ID column: skipped entirely
    - Target column: Label Encoded separately as y

    If filename and processed_folder are given, saves everything needed
    to apply this exact same encoding to a new, unlabeled file later:
    {filename}_encoders.joblib
    """

    df = df.copy()
    encoding_report = {
        "label_encoded": [],
        "one_hot_encoded": [],
        "skipped": [],
        "target": target_column
    }

    # Detect ID column if not provided
    if id_column is None:
        for col in df.columns:
            if df[col].nunique() == len(df):
                id_column = col
                break

    # Separate target column before encoding
    if target_column in df.columns:
        le_target = LabelEncoder()
        y = le_target.fit_transform(df[target_column].astype(str))
        df = df.drop(columns=[target_column])
        encoding_report["target_classes"] = list(le_target.classes_)
    else:
        y = None
        le_target = None
        encoding_report["target_classes"] = []

    # Skip ID column
    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])
        encoding_report["skipped"].append(id_column)

    # Identify column types
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    binary_cols = []
    multi_cols = []

    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= 2:
            binary_cols.append(col)
        else:
            multi_cols.append(col)

    # Label Encode binary columns -- ONE encoder PER column, each saved
    binary_encoders = {}
    for col in binary_cols:
        col_encoder = LabelEncoder()
        df[col] = col_encoder.fit_transform(df[col].astype(str))
        binary_encoders[col] = col_encoder
        encoding_report["label_encoded"].append(col)

    # One Hot Encode multi-category columns
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=False)
        encoding_report["one_hot_encoded"] = multi_cols

    # Convert all boolean columns from get_dummies to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Save everything needed to repeat this exact encoding on new data
    if filename and processed_folder:
        bundle = {
            "id_column": id_column,
            "target_column": target_column,
            "target_encoder": le_target,
            "binary_cols": binary_cols,
            "binary_encoders": binary_encoders,
            "multi_cols": multi_cols,
            "final_columns": list(df.columns),  # exact column order/set
        }
        bundle_path = os.path.join(
            processed_folder, f"{filename}_encoders.joblib"
        )
        joblib.dump(bundle, bundle_path)

    return df, y, encoding_report