import os
import json
import numpy as np
import pandas as pd
import joblib

from quantum.model_manager import ACTIVE_PREFIX, has_active_model
from quantum.quantum_vqc import load_trained_vqc
from preprocessing.missing_values import handle_missing_values
from preprocessing.duplicates import remove_duplicates

# Anyone scoring above this is called "High" risk, and so on. These are
# plain probability cutoffs, not learned from data -- easy to adjust.
RISK_THRESHOLDS = {
    "High": 0.66,
    "Medium": 0.33,
}


class NumpyEncoder(json.JSONEncoder):
    """Lets json.dump handle numpy scalar/array types that show up in
    report dicts built from pandas operations (counts, sums, dtypes)."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ColumnMismatchError(Exception):
    """Raised when an uploaded file's columns don't match what the
    active model was trained on. Carries enough detail to show the
    user exactly what's wrong."""

    def __init__(self, missing, extra):
        self.missing = missing
        self.extra = extra
        msg = []
        if missing:
            msg.append(f"missing columns: {', '.join(missing)}")
        if extra:
            msg.append(f"unexpected columns: {', '.join(extra)}")
        super().__init__("; ".join(msg))


def _p(processed_folder, suffix, prefix=ACTIVE_PREFIX):
    return os.path.join(processed_folder, f"{prefix}_{suffix}")


def _pp(processed_folder, predict_id, suffix):
    """Path for this specific prediction run's own intermediate files
    (not the active model's files -- those use _p above)."""
    return os.path.join(processed_folder, f"predict_{predict_id}_{suffix}")


def _align_model_columns(df, bundle):
    model_columns = (
        set(bundle["binary_cols"])
        | set(bundle["multi_cols"])
        | set(bundle["final_columns"])
    )
    if bundle["id_column"]:
        model_columns.add(bundle["id_column"])

    expected_by_casefold = {name.casefold(): name for name in model_columns}
    rename = {}
    matched = set()

    for column in df.columns:
        expected = expected_by_casefold.get(str(column).casefold())
        if expected is None:
            continue
        if expected in matched:
            raise ValueError(
                f"Multiple input columns match model column '{expected}'."
            )
        matched.add(expected)
        if column != expected:
            rename[column] = expected

    return df.rename(columns=rename)


def check_columns(df, processed_folder, target_column="Churn"):
    """
    Confirms a new file has the columns the active model expects,
    before spending time trying to encode/predict on it.

    Raises ColumnMismatchError if anything doesn't line up.
    """

    if not has_active_model(processed_folder):
        raise RuntimeError(
            "No active model found. Train a model and click "
            "'Save as Active Model' before predicting."
        )

    bundle = joblib.load(_p(processed_folder, "encoders.joblib"))
    df = _align_model_columns(df, bundle)

    expected_non_target = set(bundle["binary_cols"]) | set(bundle["multi_cols"])
    df_cols = set(df.columns)

    missing = [c for c in expected_non_target if c not in df_cols]

    # A Churn column being present is not an error -- it's simply dropped
    # before prediction (handy for testing with a familiar labeled file).
    # Only genuinely missing required columns block the file.
    if missing:
        raise ColumnMismatchError(missing, [])


def _apply_saved_encoding(df, processed_folder):
    """Applies the exact encoders fit during training to a new dataframe."""

    bundle = joblib.load(_p(processed_folder, "encoders.joblib"))
    df = _align_model_columns(df.copy(), bundle)

    id_column = bundle["id_column"]
    ids = df[id_column].copy() if id_column and id_column in df.columns else None
    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])

    # Binary columns: reuse each column's own fitted LabelEncoder.
    # Unseen categories are mapped to the first known class rather than
    # raising an error, since one odd row shouldn't block the whole file.
    unseen_report = []
    for col, encoder in bundle["binary_encoders"].items():
        if col not in df.columns:
            continue
        known = set(encoder.classes_)
        values = df[col].astype(str)
        n_unseen = int((~values.isin(known)).sum())
        if n_unseen:
            unseen_report.append({"column": col, "unseen_rows": n_unseen})
        values = values.where(values.isin(known), encoder.classes_[0])
        df[col] = encoder.transform(values)

    # Multi-category columns: one-hot, then force the exact column set
    # and order training produced.
    multi_cols = [c for c in bundle["multi_cols"] if c in df.columns]
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=False)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    final_columns = bundle["final_columns"]
    df = df.reindex(columns=final_columns, fill_value=0)

    report = {
        "id_column": id_column,
        "binary_encoded": list(bundle["binary_encoders"].keys()),
        "one_hot_encoded": bundle["multi_cols"],
        "total_features_after_encoding": len(final_columns),
        "unseen_categories": unseen_report,
    }

    return df, ids, report


def preprocess_for_prediction(df, processed_folder, predict_id):
    """
    Stage 1: clean + apply the saved encoding/scaling/feature-selection
    to a new, unlabeled file -- using ONLY what was fit during training.
    Nothing here trains or refits anything.

    Saves intermediate arrays under this prediction run's own id, so a
    separate later step (predict_from_preprocessed) can run the actual
    model without redoing this work.

    Returns a report dict suitable for a "preprocessing complete"
    summary page, with a button to proceed to prediction.
    """

    if not has_active_model(processed_folder):
        raise RuntimeError(
            "No active model found. Train a model and click "
            "'Save as Active Model' before predicting."
        )

    # If a Churn column is present (e.g. testing with the labeled file),
    # drop it before preprocessing -- predictions never use it.
    dropped_target = None
    for col in df.columns:
        if col.lower() == "churn":
            dropped_target = col
            df = df.drop(columns=[col])
            break

    original_df = df.copy()
    records_before = len(df)

    # --- Clean (reuses the same cleaning steps training data went through) ---
    df, missing_report = handle_missing_values(df)
    df, duplicates_report = remove_duplicates(df)

    # --- Encode, using the saved bundle ---
    df_encoded, ids, encoding_report = _apply_saved_encoding(df, processed_folder)

    # --- Scale, using the saved StandardScaler ---
    scaler = joblib.load(_p(processed_folder, "scaler.joblib"))
    X_scaled = scaler.transform(df_encoded)

    # --- Feature selection, using the saved selected feature list ---
    with open(_p(processed_folder, "feature_selection.json")) as f:
        selection = json.load(f)
    selected_names = selection["selected_features"]

    all_names = list(df_encoded.columns)
    selected_indices = [all_names.index(name) for name in selected_names]
    X_selected = X_scaled[:, selected_indices]

    # --- Save intermediates for the next stage ---
    np.save(_pp(processed_folder, predict_id, "X_selected.npy"), X_selected)
    original_df.to_csv(_pp(processed_folder, predict_id, "original.csv"), index=False)
    if ids is not None:
        ids.to_frame(name="_id").to_csv(
            _pp(processed_folder, predict_id, "ids.csv"), index=False
        )

    preprocessing_report = {
        "predict_id": predict_id,
        "dropped_target_column": dropped_target,
        "records_before": records_before,
        "records_after": len(df),
        "missing_report": missing_report,
        "duplicates_report": duplicates_report,
        "encoding_report": encoding_report,
        "scaling": {
            "total_features_scaled": X_scaled.shape[1],
        },
        "feature_selection": {
            "selected_features": selected_names,
            "n_features": len(selected_names),
        },
    }

    with open(_pp(processed_folder, predict_id, "preprocess_report.json"), "w") as f:
        json.dump(preprocessing_report, f, cls=NumpyEncoder)

    return preprocessing_report


def predict_from_preprocessed(processed_folder, predict_id):
    """
    Stage 2: runs the active, trained VQC on data already prepared by
    preprocess_for_prediction(). This is the only stage that touches
    the quantum model -- and it only ever predicts, never trains.
    """

    X_selected = np.load(_pp(processed_folder, predict_id, "X_selected.npy"))
    original_df = pd.read_csv(_pp(processed_folder, predict_id, "original.csv"))

    ids = None
    ids_path = _pp(processed_folder, predict_id, "ids.csv")
    if os.path.exists(ids_path):
        ids = pd.read_csv(ids_path)["_id"]

    vqc_scaler = joblib.load(_p(processed_folder, "vqc_scaler.joblib"))
    X_quantum = np.clip(vqc_scaler.transform(X_selected), 0, np.pi)

    vqc = load_trained_vqc(processed_folder)
    proba = vqc.predict_proba(X_quantum)
    churn_probability = proba[:, 1]

    with open(_p(processed_folder, "vqc_metrics.json")) as f:
        active_metrics = json.load(f)
    operating_threshold = active_metrics.get("operating_threshold", 0.5)

    def risk_level(p):
        if p >= RISK_THRESHOLDS["High"]:
            return "High"
        if p >= RISK_THRESHOLDS["Medium"]:
            return "Medium"
        return "Low"

    result = original_df.copy()
    result["churn_probability"] = np.round(churn_probability, 4)
    result["predicted_churn"] = [
        "Yes" if p >= operating_threshold else "No" for p in churn_probability
    ]
    result["risk_level"] = [risk_level(p) for p in churn_probability]

    lead_cols = []
    if ids is not None:
        result[ids.name if hasattr(ids, "name") else "_id"] = ids.values
        lead_cols.append(ids.name if hasattr(ids, "name") else "_id")
    lead_cols += ["churn_probability", "predicted_churn", "risk_level"]
    other_cols = [c for c in result.columns if c not in lead_cols]
    result = result[lead_cols + other_cols]

    result = result.sort_values("churn_probability", ascending=False)
    result = result.reset_index(drop=True)

    return result


def summarize_predictions(result_df):
    """Small summary dict for the results page header."""

    total = len(result_df)
    counts = result_df["risk_level"].value_counts().to_dict()
    return {
        "total_customers": int(total),
        "high_risk": int(counts.get("High", 0)),
        "medium_risk": int(counts.get("Medium", 0)),
        "low_risk": int(counts.get("Low", 0)),
        "high_risk_pct": round(100 * counts.get("High", 0) / total, 1) if total else 0,
        "average_probability": round(float(result_df["churn_probability"].mean()), 4) if total else 0,
    }


# ---------------------------------------------------------------------
# Convenience: both stages in one call, for callers that don't need the
# intermediate review page.
# ---------------------------------------------------------------------
def predict_new_customers(df, processed_folder, predict_id="oneshot"):
    preprocess_for_prediction(df, processed_folder, predict_id)
    return predict_from_preprocessed(processed_folder, predict_id)