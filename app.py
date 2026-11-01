import io
import hmac
import os
import re
import json
import shutil
import numpy as np

from flask import (
    Flask, render_template, request,
    redirect, flash, jsonify, send_file
)

from report.report_builder import build_report_xlsx

import pandas as pd
from segmentation.segmenter import assign_segments, segment_summary, SEGMENTS
from preprocessing.loader import load_dataset
from preprocessing.summary import dataset_summary
from preprocessing.missing_values import handle_missing_values
from preprocessing.duplicates import remove_duplicates
from preprocessing.encoding import encode_features
from preprocessing.scaling import scale_features
from retention.retention_engine import get_retention_strategies
from preprocessing.outliers import (
    detect_outliers,
    detect_negative_values,
    detect_missing_id_rows
)
from preprocessing.ai_recommendation import (
    get_recommendations,
    get_negative_recommendations
)
from preprocessing.outlier_actions import apply_actions

from quantum.feature_selection import select_features
from quantum.quantum_vqc import (
    start_vqc_training, read_vqc_status, fast_backend_available,
    evaluate_at_thresholds, set_operating_threshold, repair_missing_dill
)
from quantum.model_manager import (
    activate_model, has_active_model, load_active_model_info
)

from predict.predict_pipeline import (
    predict_new_customers, preprocess_for_prediction,
    predict_from_preprocessed, summarize_predictions,
    check_columns, ColumnMismatchError
)

from config import (
    UPLOAD_FOLDER,
    PROCESSED_FOLDER,
    INITIAL_PROCESSED_FOLDER,
    MAX_CONTENT_LENGTH
)

from utils.validation import allowed_file

app = Flask(__name__)

app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Seed Vercel's writable temp storage with the bundled default model files.
if os.getenv("VERCEL") == "1" and not os.listdir(PROCESSED_FOLDER):
    if os.path.isdir(INITIAL_PROCESSED_FOLDER):
        shutil.copytree(
            INITIAL_PROCESSED_FOLDER, PROCESSED_FOLDER, dirs_exist_ok=True
        )


@app.before_request
def require_app_login():
    if request.endpoint in {"healthcheck", "static"}:
        return None

    if os.getenv("APP_AUTH_REQUIRED", "false").lower() != "true":
        return None

    username = os.getenv("APP_USERNAME")
    password = os.getenv("APP_PASSWORD")
    if not username or not password:
        return "Application access is not configured.", 503

    credentials = request.authorization
    if (credentials is None
            or not hmac.compare_digest(credentials.username or "", username)
            or not hmac.compare_digest(credentials.password or "", password)):
        response = app.response_class("Authentication required.", 401)
        response.headers["WWW-Authenticate"] = 'Basic realm="ChurnQ"'
        return response


@app.route("/health")
def healthcheck():
    return jsonify(status="ok")


def generate_filename(original_filename):

    extension = original_filename.rsplit(".", 1)[1].lower()
    existing_files = os.listdir(app.config["UPLOAD_FOLDER"])
    numbers = []
    pattern = rf"Telco dataset-(\d+)\.{extension}"

    for file in existing_files:
        match = re.match(pattern, file)
        if match:
            numbers.append(int(match.group(1)))

    next_number = max(numbers) + 1 if numbers else 1
    return f"Telco dataset-{next_number}.{extension}"


# ---------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------------------
# Modules 1-3: Upload, clean, summarize
# ---------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_file():

    if "dataset" not in request.files:
        flash("No file selected.")
        return redirect("/")

    file = request.files["dataset"]

    if file.filename == "":
        flash("Please choose a CSV, Excel, or OpenDocument file.")
        return redirect("/")

    if file and allowed_file(file.filename):

        filename = generate_filename(file.filename)
        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"], filename
        )
        file.save(filepath)

        try:
            df = load_dataset(filepath)
        except ValueError as e:
            flash(str(e))
            return redirect("/")

        summary = dataset_summary(df)
        df, missing_report = handle_missing_values(df)
        df, duplicates_report = remove_duplicates(df)

        # Save cleaned dataframe for anomaly review
        cleaned_path = os.path.join(
            app.config["PROCESSED_FOLDER"],
            f"{filename}_cleaned.csv"
        )
        df.to_csv(cleaned_path, index=False)

        # fallback_count belongs to the retention page (Module 19).
        # It is passed here as 0 so result.html still renders if the
        # template references it (e.g. a copied retention block).
        return render_template(
            "result.html",
            filename=filename,
            summary=summary,
            missing_report=missing_report,
            duplicates_report=duplicates_report,
            fallback_count=0,
            preview=df.head(10).to_html(
                classes="table",
                index=False
            )
        )

    flash("Only CSV, Excel, or OpenDocument files are allowed.")
    return redirect("/")


# ---------------------------------------------------------------------
# Modules 6-11: Anomaly review and cleanup actions
# ---------------------------------------------------------------------
@app.route("/anomaly-review/<filename>")
def anomaly_review(filename):

    cleaned_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_cleaned.csv"
    )

    if not os.path.exists(cleaned_path):
        flash("Session expired. Please upload again.")
        return redirect("/")

    df = load_dataset(cleaned_path)
    anomaly_report = detect_outliers(df)
    negative_report = detect_negative_values(df)
    missing_id_report = detect_missing_id_rows(df)
    recommendations = get_recommendations(anomaly_report)
    negative_recommendations = get_negative_recommendations(
        negative_report
    )

    # Cache recommendations to avoid re-calling Gemini
    cache = {
        "recommendations": recommendations,
        "negative_recommendations": negative_recommendations
    }
    cache_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_recommendations.json"
    )
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    return render_template(
        "anomaly_review.html",
        filename=filename,
        total_records=len(df),
        anomaly_report=anomaly_report,
        recommendations=recommendations,
        negative_report=negative_report,
        negative_recommendations=negative_recommendations,
        missing_id_report=missing_id_report
    )


@app.route("/apply-actions/<filename>", methods=["POST"])
def apply_actions_route(filename):

    cleaned_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_cleaned.csv"
    )

    cache_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_recommendations.json"
    )

    if not os.path.exists(cleaned_path):
        flash("Session expired. Please upload again.")
        return redirect("/")

    df = load_dataset(cleaned_path)
    anomaly_report = detect_outliers(df)
    negative_report = detect_negative_values(df)
    missing_id_report = detect_missing_id_rows(df)

    # Load cached recommendations
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
        recommendations = cache["recommendations"]
        negative_recommendations = cache["negative_recommendations"]
    else:
        recommendations = get_recommendations(anomaly_report)
        negative_recommendations = get_negative_recommendations(
            negative_report
        )

    # Collect column-level anomaly actions
    actions = {}
    for col in anomaly_report.keys():
        action = request.form.get(f"action_{col}", "Retain Records")
        if action == "Apply AI Recommendation":
            action = recommendations[col]["recommendation"]
        actions[col] = action

    # Collect per-column negative actions
    negative_actions = {}
    for col in negative_report.get("columns", {}).keys():
        neg_action = request.form.get(
            f"neg_action_{col}", "Replace with Median"
        )
        if neg_action == "Apply AI Recommendation":
            neg_action = negative_recommendations[col]["recommendation"]
        negative_actions[col] = neg_action

    # Collect retained missing ID rows
    retained_missing_id_rows = request.form.getlist("retain_missing_id")

    df, preprocessing_log = apply_actions(
        df, anomaly_report, actions,
        negative_report, negative_actions,
        missing_id_report, retained_missing_id_rows
    )

    # Get retained missing ID rows for display
    retained_missing_data = []
    for idx_str in retained_missing_id_rows:
        idx = int(idx_str)
        if idx in df.index:
            row = df.loc[idx].to_dict()
            placeholder = missing_id_report[idx_str]["placeholder_id"]
            row["_placeholder_id"] = placeholder
            retained_missing_data.append(row)

    final_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_processed.csv"
    )
    df.to_csv(final_path, index=False)

    # Clean up cache
    if os.path.exists(cache_path):
        os.remove(cache_path)

    # The model is trained once: with an active model, the next step is
    # always prediction, whether or not the file has a Churn column.
    # has_churn only tells the template whether retraining is possible.
    has_churn = any(c.lower() == "churn" for c in df.columns)
    use_active_model = has_active_model(app.config["PROCESSED_FOLDER"])

    return render_template(
        "preprocessing_complete.html",
        filename=filename,
        use_active_model=use_active_model,
        has_churn=has_churn,
        total_records=len(df),
        total_columns=len(df.columns),
        actions_applied=actions,
        negative_actions_applied=negative_actions,
        preprocessing_log=preprocessing_log,
        retained_missing_data=retained_missing_data,
        preview=df.head(10).to_html(
            classes="table",
            index=False
        )
    )


# ---------------------------------------------------------------------
# Module 12: Encoding
# ---------------------------------------------------------------------
@app.route("/encode/<filename>")
def encode(filename):

    processed_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_processed.csv"
    )

    if not os.path.exists(processed_path):
        flash("Session expired. Please upload again.")
        return redirect("/")

    # The model is trained once. With an active model, every upload goes
    # to prediction (a Churn column, if present, is ignored). Retraining
    # only happens when explicitly requested via /encode/<file>?train=1
    if (has_active_model(app.config["PROCESSED_FOLDER"])
            and request.args.get("train") != "1"):
        return redirect(f"/predict-active/{filename}")

    df = load_dataset(processed_path)

    # Detect target and ID columns
    target_column = None
    for col in df.columns:
        if col.lower() == "churn":
            target_column = col
            break

    if target_column is None:
        flash("Training needs a labeled file: Churn column not found.")
        return redirect("/")

    df_encoded, y, encoding_report = encode_features(
        df, target_column=target_column,
        filename=filename, processed_folder=app.config["PROCESSED_FOLDER"]
    )

    # Save encoded dataframe
    encoded_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_encoded.csv"
    )
    df_encoded.to_csv(encoded_path, index=False)

    # Save target variable
    target_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_target.npy"
    )
    np.save(target_path, y)

    return render_template(
        "encoding_complete.html",
        filename=filename,
        total_records=len(df_encoded),
        total_features=len(df_encoded.columns),
        encoding_report=encoding_report,
        preview=df_encoded.head(10).to_html(
            classes="table",
            index=False
        )
    )


# ---------------------------------------------------------------------
# Module 13: Scaling
# ---------------------------------------------------------------------
@app.route("/scale/<filename>")
def scale(filename):

    encoded_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_encoded.csv"
    )

    target_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_target.npy"
    )

    if not os.path.exists(encoded_path):
        flash("Session expired. Please upload again.")
        return redirect("/")

    df_encoded = load_dataset(encoded_path)
    y = np.load(target_path)

    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaling_report = scale_features(
        df_encoded, y,
        filename,
        app.config["PROCESSED_FOLDER"]
    )

    # Save scaled data
    np.save(os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_X_train.npy"
    ), X_train_scaled)

    np.save(os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_X_test.npy"
    ), X_test_scaled)

    np.save(os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_y_train.npy"
    ), y_train)

    np.save(os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_y_test.npy"
    ), y_test)

    # Save feature names
    with open(os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_feature_names.json"
    ), "w") as f:
        json.dump(feature_names, f)

    return render_template(
        "scaling_complete.html",
        filename=filename,
        scaling_report=scaling_report
    )


# ---------------------------------------------------------------------
# Module 14: Feature selection
# ---------------------------------------------------------------------
@app.route("/feature-selection/<filename>")
def feature_selection(filename):

    X_train_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_X_train.npy"
    )
    y_train_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_y_train.npy"
    )
    feature_names_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_feature_names.json"
    )

    if not os.path.exists(X_train_path):
        flash("Session expired. Please upload again.")
        return redirect("/")

    X_train_scaled = np.load(X_train_path)
    y_train = np.load(y_train_path)

    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    X_train_selected, selected_feature_names, selection_report = select_features(
        X_train_scaled, y_train,
        feature_names, filename,
        app.config["PROCESSED_FOLDER"]
    )

    # Save selected training data
    np.save(os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_X_train_selected.npy"
    ), X_train_selected)

    # Apply same selection to test set
    X_test_scaled = np.load(os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_X_test.npy"
    ))

    selected_indices = [
        feature_names.index(name)
        for name in selected_feature_names
    ]
    X_test_selected = X_test_scaled[:, selected_indices]

    np.save(os.path.join(
        app.config["PROCESSED_FOLDER"],
        f"{filename}_X_test_selected.npy"
    ), X_test_selected)

    return render_template(
        "feature_selection_complete.html",
        filename=filename,
        selection_report=selection_report
    )


# ---------------------------------------------------------------------
# Module 16: Quantum VQC training
# ---------------------------------------------------------------------
@app.route("/predict/<filename>")
def predict(filename):

    folder = app.config["PROCESSED_FOLDER"]

    def p(suffix):
        return os.path.join(folder, f"{filename}_{suffix}")

    if not os.path.exists(p("X_train_selected.npy")):
        flash("Session expired. Please upload again.")
        return redirect("/")

    y_train = np.load(p("y_train.npy")).astype(int)
    max_train = 2 * int(min((y_train == 1).sum(), (y_train == 0).sum()))
    full_train = int(len(y_train))

    status = read_vqc_status(filename, folder)

    vqc_report = None
    if status["state"] != "running" and os.path.exists(p("vqc_metrics.json")):
        with open(p("vqc_metrics.json"), "r") as f:
            vqc_report = json.load(f)

    is_active = False
    if vqc_report is not None:
        active_info = None
        if has_active_model(folder):
            with open(os.path.join(folder, "active_model_vqc_metrics.json")) as f:
                active_info = json.load(f)
        is_active = (
            active_info is not None
            and active_info.get("_trained_from_file") == filename
        )

    return render_template(
        "prediction_results.html",
        filename=filename,
        vqc=vqc_report,
        status=status,
        max_train=max_train,
        full_train=full_train,
        aer_available=fast_backend_available(),
        is_active=is_active,
        threshold_results=None,
        current_threshold=vqc_report.get("operating_threshold") if vqc_report else None
    )


@app.route("/train-vqc/<filename>", methods=["POST"])
def train_vqc(filename):

    folder = app.config["PROCESSED_FOLDER"]

    def p(suffix):
        return os.path.join(folder, f"{filename}_{suffix}")

    if not os.path.exists(p("X_train_selected.npy")):
        flash("Session expired. Please upload again.")
        return redirect("/")

    balance = request.form.get("balance", "balanced")
    try:
        maxiter = int(request.form.get("maxiter", 100))
    except ValueError:
        maxiter = 100

    backend = request.form.get("backend", "fast")

    with open(p("feature_selection.json"), "r") as f:
        selected_names = json.load(f)["selected_features"]

    started = start_vqc_training(
        np.load(p("X_train_selected.npy")),
        np.load(p("y_train.npy")),
        np.load(p("X_test_selected.npy")),
        np.load(p("y_test.npy")),
        selected_names, filename, folder,
        n_train=0, maxiter=maxiter, backend=backend, balance=balance
    )

    if not started:
        flash("Training is already running for this dataset.")

    return redirect(f"/predict/{filename}")


@app.route("/vqc-status/<filename>")
def vqc_status(filename):
    return jsonify(
        read_vqc_status(filename, app.config["PROCESSED_FOLDER"])
    )


@app.route("/repair-model/<filename>")
def repair_model(filename):
    folder = app.config["PROCESSED_FOLDER"]
    repair_missing_dill(folder, filename)
    flash("Model repaired — you can now use Cutoff Comparison.")
    return redirect(f"/predict/{filename}")


@app.route("/threshold-analysis/<filename>", methods=["POST"])
def threshold_analysis(filename):

    folder = app.config["PROCESSED_FOLDER"]
    results = evaluate_at_thresholds(folder, filename)

    with open(os.path.join(folder, f"{filename}_vqc_metrics.json")) as f:
        vqc_report = json.load(f)

    y_train = np.load(os.path.join(folder, f"{filename}_y_train.npy")).astype(int)
    max_train = 2 * int(min((y_train == 1).sum(), (y_train == 0).sum()))
    full_train = int(len(y_train))

    return render_template(
        "prediction_results.html",
        filename=filename,
        vqc=vqc_report,
        status={"state": "done"},
        max_train=max_train,
        full_train=full_train,
        aer_available=fast_backend_available(),
        is_active=False,
        threshold_results=results,
        current_threshold=vqc_report.get("operating_threshold")
    )


@app.route("/set-threshold/<filename>", methods=["POST"])
def set_threshold(filename):
    folder = app.config["PROCESSED_FOLDER"]
    threshold = float(request.form.get("threshold", 0.5))
    set_operating_threshold(folder, filename, threshold)
    flash(f"Cutoff set to {threshold}. This will be used when predicting new customers.")
    return redirect(f"/predict/{filename}")


@app.route("/activate-model/<filename>", methods=["POST"])
def activate_model_route(filename):

    folder = app.config["PROCESSED_FOLDER"]
    success, missing = activate_model(filename, folder)

    if success:
        flash("This model is now the active model for predictions.")
    else:
        flash(
            f"Could not activate: missing {missing}. "
            "Make sure training finished and encoding was re-run "
            "after the encoders.joblib fix."
        )

    return redirect(f"/predict/{filename}")


# ---------------------------------------------------------------------
# Module 17: Predict churn for new, unlabeled customers
# ---------------------------------------------------------------------
@app.route("/predict-new")
def predict_new_upload():
    folder = app.config["PROCESSED_FOLDER"]
    return render_template(
        "predict_upload.html",
        has_model=has_active_model(folder),
        active_info=load_active_model_info(folder)
    )


@app.route("/predict-new/upload", methods=["POST"])
def predict_new_upload_post():

    folder = app.config["PROCESSED_FOLDER"]

    if "dataset" not in request.files or request.files["dataset"].filename == "":
        flash("Please choose a file.")
        return redirect("/predict-new")

    file = request.files["dataset"]
    if not allowed_file(file.filename):
        flash("Only CSV, Excel, or OpenDocument files are allowed.")
        return redirect("/predict-new")

    filename = generate_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"predict_{filename}")
    file.save(filepath)

    try:
        df = load_dataset(filepath)
        check_columns(df, folder)
        report = preprocess_for_prediction(df, folder, filename)
    except ColumnMismatchError as e:
        flash(f"This file doesn't match the training data: {e}")
        return redirect("/predict-new")
    except Exception as e:
        flash(f"Preprocessing failed: {e}")
        return redirect("/predict-new")

    return render_template(
        "predict_preprocessing_complete.html",
        filename=filename,
        report=report
    )


@app.route("/predict-new/run/<predict_id>")
def predict_new_run(predict_id):

    folder = app.config["PROCESSED_FOLDER"]

    try:
        result = predict_from_preprocessed(folder, predict_id)
    except Exception as e:
        flash(f"Prediction failed: {e}")
        return redirect("/predict-new")

    result.to_csv(os.path.join(folder, f"{predict_id}_predictions.csv"), index=False)

    return render_template(
        "predict_output.html",
        filename=predict_id,
        summary=summarize_predictions(result),
        preview=result.head(50).to_html(classes="table", index=False)
    )


@app.route("/predict-new/download/<filename>")
def predict_new_download(filename):
    path = os.path.join(
        app.config["PROCESSED_FOLDER"], f"{filename}_predictions.csv"
    )
    return send_file(
        path, as_attachment=True,
        download_name=f"churn_predictions_{filename}.csv"
    )


@app.route("/predict-active/<filename>")
def predict_active(filename):

    folder = app.config["PROCESSED_FOLDER"]
    processed_path = os.path.join(folder, f"{filename}_processed.csv")

    if not os.path.exists(processed_path):
        flash("Session expired. Please upload again.")
        return redirect("/")

    df = load_dataset(processed_path)

    try:
        check_columns(df, folder)
        report = preprocess_for_prediction(df, folder, filename)
    except ColumnMismatchError as e:
        flash(f"This file doesn't match the active model: {e}")
        return redirect("/")
    except Exception as e:
        flash(f"Preprocessing failed: {e}")
        return redirect("/")

    return render_template(
        "predict_preprocessing_complete.html",
        filename=filename,
        report=report
    )


# ---------------------------------------------------------------------
# Module 18: Behavioral segmentation
# ---------------------------------------------------------------------
SEGMENT_REQUIRED_COLS = [
    "tenure", "MonthlyCharges", "Contract", "predicted_churn", "risk_level"
]


def _churn_to_yes_no(value):
    """predicted_churn may be stored as Yes/No, 1/0 or True/False."""
    s = str(value).strip().lower()
    return "Yes" if s in ("yes", "1", "1.0", "true") else "No"


def _load_segmented(filename):
    """Load saved predictions and assign segments.
    Returns (dataframe, None) on success or (None, error_message)."""

    path = os.path.join(
        app.config["PROCESSED_FOLDER"], f"{filename}_predictions.csv"
    )

    if not os.path.exists(path):
        return None, (
            "No prediction results found for this file. "
            "Please run the prediction first."
        )

    df = pd.read_csv(path)

    # Match the required columns case-insensitively
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for needed in SEGMENT_REQUIRED_COLS:
        actual = lower_map.get(needed.lower())
        if actual is not None and actual != needed:
            rename[actual] = needed
    if rename:
        df = df.rename(columns=rename)

    missing = [c for c in SEGMENT_REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, (
            "The saved predictions file is missing columns needed for "
            f"segmentation: {', '.join(missing)}."
        )

    if pd.api.types.is_numeric_dtype(df["Contract"]):
        return None, (
            "The Contract column in the predictions file holds encoded "
            "numbers, not the original contract names. Segmentation needs "
            "the original (un-encoded) customer columns."
        )

    df["predicted_churn"] = df["predicted_churn"].map(_churn_to_yes_no)
    df["risk_level"] = (
        df["risk_level"].astype(str)
        .str.replace(" risk", "", case=False, regex=False)
        .str.strip().str.title()
    )

    return assign_segments(df), None


@app.route("/segmentation/<filename>")
def segmentation(filename):

    df, error = _load_segmented(filename)
    if error:
        flash(error)
        return redirect("/predict-new")

    total = len(df)
    counts, by_risk = segment_summary(df)

    segment_cards = [
        {
            "name": s,
            "count": int(counts[s]),
            "pct": round(100 * int(counts[s]) / total, 1) if total else 0
        }
        for s in SEGMENTS
    ]

    risk_table = []
    for s in SEGMENTS:
        row = {"name": s, "total": int(counts[s])}
        for level in ("High", "Medium", "Low"):
            row[level] = (
                int(by_risk.loc[s, level]) if level in by_risk.columns else 0
            )
        risk_table.append(row)

    # Customer table: ID, segment, key drivers, prediction
    id_col = next(
        (c for c in df.columns if c.lower() in ("customerid", "customer_id")),
        None
    )
    wanted = [
        id_col, "segment", "tenure", "MonthlyCharges", "Contract",
        "num_services", "churn_probability", "predicted_churn", "risk_level"
    ]
    view_cols = [c for c in wanted if c is not None and c in df.columns]
    view = df[view_cols].copy()

    if "churn_probability" in view.columns:
        view = view.sort_values("churn_probability", ascending=False)
        view["churn_probability"] = view["churn_probability"].round(3)

    view = view.rename(columns={"segment": "Segment"})

    return render_template(
        "segmentation.html",
        filename=filename,
        total=total,
        segment_cards=segment_cards,
        risk_table=risk_table,
        segments=SEGMENTS,
        columns=list(view.columns),
        rows=view.to_dict("records")
    )


@app.route("/segmentation/download/<filename>")
def segmentation_download(filename):

    df, error = _load_segmented(filename)
    if error:
        flash(error)
        return redirect("/predict-new")

    out_path = os.path.join(
        app.config["PROCESSED_FOLDER"], f"{filename}_segments.csv"
    )
    df.rename(columns={"segment": "Segment"}).to_csv(out_path, index=False)

    return send_file(
        out_path, as_attachment=True,
        download_name=f"customer_segments_{filename}.csv"
    )


# ---------------------------------------------------------------------
# Module 20: Interactive dashboard
# ---------------------------------------------------------------------
DASHBOARD_MAX_ROWS = 50000


def _dashboard_payload(df, limit=DASHBOARD_MAX_ROWS):
    """Row-level data for the interactive dashboard, in a compact
    column-wise layout. All filtering and charting happens in the
    browser, so every chart can respond to every filter."""

    df = df.copy()
    total_rows = len(df)
    has_prob = "churn_probability" in df.columns
    if has_prob:
        df["churn_probability"] = pd.to_numeric(
            df["churn_probability"], errors="coerce"
        ).fillna(0)

    truncated = False
    if total_rows > limit:
        df = df.sort_values(
            "churn_probability" if has_prob else "MonthlyCharges",
            ascending=False
        ).head(limit)
        truncated = True

    id_col = next(
        (c for c in df.columns if c.lower() in ("customerid", "customer_id")),
        None
    )
    n = len(df)

    cols = {
        "id": (df[id_col].astype(str).tolist() if id_col
               else [str(i + 1) for i in range(n)]),
        "seg": df["segment"].astype(str).tolist(),
        "risk": df["risk_level"].astype(str).tolist(),
        "contract": df["Contract"].astype(str).tolist(),
        "tenure": [int(v) for v in df["tenure"]],
        "charge": [round(float(v), 2) for v in df["MonthlyCharges"]],
        "services": [int(v) for v in df["num_services"]],
        "prob": ([round(float(v), 4) for v in df["churn_probability"]]
                 if has_prob else None),
        "churn": [1 if v == "Yes" else 0 for v in df["predicted_churn"]],
    }

    return {
        "cols": cols,
        "has_prob": has_prob,
        "segments": list(SEGMENTS),
        "total_rows": int(total_rows),
        "truncated": truncated,
        "limit": limit,
    }


@app.route("/dashboard/<filename>")
def dashboard(filename):

    df, error = _load_segmented(filename)
    if error:
        flash(error)
        return redirect("/predict-new")

    return render_template(
        "dashboard.html",
        filename=filename,
        payload=_dashboard_payload(df)
    )


# ---------------------------------------------------------------------
# Module 19: Retention strategy (Gemini)
# ---------------------------------------------------------------------
def _retention_page(filename, force):

    df, error = _load_segmented(filename)
    if error:
        flash(error)
        return redirect("/predict-new")

    cache_path = os.path.join(
        app.config["PROCESSED_FOLDER"], f"{filename}_retention.json"
    )
    data = get_retention_strategies(df, SEGMENTS, cache_path, force=force)

    return render_template(
        "retention.html",
        filename=filename,
        strategies=data["strategies"],
        fallback_count=data.get("fallback_count", 0)
    )


@app.route("/retention/<filename>")
def retention(filename):
    return _retention_page(filename, force=False)


@app.route("/retention/regenerate/<filename>", methods=["POST"])
def retention_regenerate(filename):
    return _retention_page(filename, force=True)


# ---------------------------------------------------------------------
# Download: one Excel report (dashboard, retention strategy, customer list)
# ---------------------------------------------------------------------
@app.route("/report/download/<filename>")
def download_full_report(filename):

    df, error = _load_segmented(filename)
    if error:
        flash(error)
        return redirect("/predict-new")

    # Use saved retention strategies; if the retention page was never opened
    # for this file, create them now (rule-based fallback if Gemini fails).
    cache_path = os.path.join(
        app.config["PROCESSED_FOLDER"], f"{filename}_retention.json"
    )
    try:
        strategies = get_retention_strategies(
            df, SEGMENTS, cache_path, force=False
        )["strategies"]
    except Exception:
        strategies = None

    try:
        data, name = build_report_xlsx(
            app.config["PROCESSED_FOLDER"], filename,
            df=df, retention=strategies
        )
    except Exception as e:
        flash(f"Could not create the report: {e}")
        return redirect("/predict-new")

    return send_file(
        io.BytesIO(data),
        mimetype=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        as_attachment=True,
        download_name=name
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
