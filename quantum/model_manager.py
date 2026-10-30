import os
import shutil
import json

# The fixed prefix every prediction-time lookup uses, regardless of
# which uploaded filename the model was originally trained on.
ACTIVE_PREFIX = "active_model"

# (source suffix as saved during training, destination suffix under
# the active model prefix)
ARTIFACT_SUFFIXES = [
    "encoders.joblib",
    "scaler.joblib",
    "feature_selection.json",
    "vqc_weights.npy",
    "vqc_scaler.joblib",
    "vqc_metrics.json",
    "vqc_model.dill",
]


def _src(processed_folder, filename, suffix):
    return os.path.join(processed_folder, f"{filename}_{suffix}")


def _dst(processed_folder, suffix):
    return os.path.join(processed_folder, f"{ACTIVE_PREFIX}_{suffix}")


def activate_model(filename, processed_folder):
    """
    Promotes the artifacts trained on `filename` to be THE active model,
    saved under a fixed name independent of any uploaded filename.

    From this point on, any file uploaded for prediction -- named
    anything at all -- uses these active_model_* files, never the
    original training filename's own copies.

    Returns (True, None) on success, or (False, missing_suffix) if a
    required training artifact is missing (e.g. training never
    finished, or was run before the encoding.py fix was applied).
    """

    for suffix in ARTIFACT_SUFFIXES:
        src = _src(processed_folder, filename, suffix)
        if not os.path.exists(src):
            return False, suffix

    for suffix in ARTIFACT_SUFFIXES:
        shutil.copy2(
            _src(processed_folder, filename, suffix),
            _dst(processed_folder, suffix)
        )

    # Record which upload this active model came from, for reference
    with open(_dst(processed_folder, "vqc_metrics.json"), "r") as f:
        metrics = json.load(f)
    metrics["_trained_from_file"] = filename
    with open(_dst(processed_folder, "vqc_metrics.json"), "w") as f:
        json.dump(metrics, f)

    return True, None


def has_active_model(processed_folder):
    return all(
        os.path.exists(_dst(processed_folder, suffix))
        for suffix in ARTIFACT_SUFFIXES
    )


def get_active_model_paths(processed_folder):
    """Returns a dict of {short_name: full_path} for the active model files."""
    return {
        suffix.split(".")[0]: _dst(processed_folder, suffix)
        for suffix in ARTIFACT_SUFFIXES
    }


def load_active_model_info(processed_folder):
    """Returns the active model's saved metrics dict, or None if not active."""
    if not has_active_model(processed_folder):
        return None
    with open(_dst(processed_folder, "vqc_metrics.json"), "r") as f:
        return json.load(f)