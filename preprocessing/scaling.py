import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


def scale_features(df_encoded, y, filename, processed_folder):
    """
    Applies StandardScaler to all numeric features.

    Steps:
    1. Split data into train and test sets (80/20)
    2. Fit scaler on training data only (prevents data leakage)
    3. Transform both train and test sets
    4. Save scaler to disk for later use in prediction

    Returns:
    - X_train_scaled, X_test_scaled, y_train, y_test
    - feature_names (column names after encoding)
    - scaling_report
    """

    feature_names = list(df_encoded.columns)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_encoded, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # preserve class imbalance ratio in both splits
    )

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler to disk
    scaler_path = os.path.join(
        processed_folder,
        f"{filename}_scaler.joblib"
    )
    joblib.dump(scaler, scaler_path)

    scaling_report = {
        "total_features": len(feature_names),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_percentage": round(len(X_train) / len(df_encoded) * 100, 1),
        "test_percentage": round(len(X_test) / len(df_encoded) * 100, 1),
        "feature_names": feature_names
    }

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_names,
        scaling_report
    )