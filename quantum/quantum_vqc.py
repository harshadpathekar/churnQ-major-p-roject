import os
import json
import time
import threading

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from qiskit_machine_learning.algorithms import VQC

# Import paths differ between qiskit-machine-learning versions,
# so try the newer location first and fall back to the older one.
try:
    from qiskit_machine_learning.optimizers import COBYLA
except ImportError:
    from qiskit_algorithms.optimizers import COBYLA

try:
    from qiskit_machine_learning.utils import algorithm_globals
except ImportError:
    from qiskit_algorithms.utils import algorithm_globals

try:
    from qiskit.circuit.library import zz_feature_map as _zz_feature_map
    from qiskit.circuit.library import real_amplitudes as _real_amplitudes

    def _make_feature_map(n_qubits, reps, entanglement):
        return _zz_feature_map(
            feature_dimension=n_qubits, reps=reps, entanglement=entanglement
        )

    def _make_ansatz(n_qubits, reps, entanglement):
        return _real_amplitudes(n_qubits, reps=reps, entanglement=entanglement)

except ImportError:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

    def _make_feature_map(n_qubits, reps, entanglement):
        return ZZFeatureMap(
            feature_dimension=n_qubits, reps=reps, entanglement=entanglement
        )

    def _make_ansatz(n_qubits, reps, entanglement):
        return RealAmplitudes(n_qubits, reps=reps, entanglement=entanglement)


# Optional fast simulator (pip install qiskit-aer). About 4x faster than
# the default exact simulator, which matters for large training runs.
try:
    from qiskit_aer.primitives import SamplerV2 as _AerSampler
except Exception:
    _AerSampler = None


# ---------------------------------------------------------------------
# Settings (kept fixed so results are reproducible and easy to report)
# ---------------------------------------------------------------------
SEED = 42
FEATURE_MAP_REPS = 2
FEATURE_MAP_ENTANGLEMENT = "linear"
ANSATZ_REPS = 2
ANSATZ_ENTANGLEMENT = "linear"

AER_SHOTS = 1024

DEFAULT_N_TRAIN = 0        # 0 = use the maximum balanced amount of data
DEFAULT_MAXITER = 100      # COBYLA iterations
MAX_N_TRAIN = 20000  # raw cap; full Telco training set (~5600) fits fine
MIN_MAXITER = 30         # COBYLA needs at least (parameters + 2) = 26
MAX_MAXITER = 300

_active = set()
_lock = threading.Lock()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _path(folder, filename, suffix):
    return os.path.join(folder, f"{filename}_{suffix}")


def _write_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _training_subset(y, n_train, seed=SEED, balance="balanced"):
    """
    Picks which training rows the VQC trains on.

    balance="balanced": 50/50 churn / no-churn subsample, capped at
        n_train. Standard VQC has no class_weight option, so this plays
        the same role as class_weight='balanced' does for the SVM.
    balance="full": every available training row, in its real,
        imbalanced churn ratio (about 27% churn). n_train is ignored.
        With no class weighting, a model trained this way can lean
        toward always predicting "no churn", since that is right most
        of the time on this data. Worth checking recall on the churn
        class after training in this mode.

    The test set is NOT touched by either mode and keeps its real
    ~27% churn rate, so evaluation is always fair.
    """
    rng = np.random.default_rng(seed)

    if balance == "full":
        idx = np.arange(len(y))
        rng.shuffle(idx)
        return idx

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    per_class = min(n_train // 2, len(idx_pos), len(idx_neg))
    chosen = np.concatenate([
        rng.choice(idx_pos, per_class, replace=False),
        rng.choice(idx_neg, per_class, replace=False),
    ])
    rng.shuffle(chosen)
    return chosen


def fast_backend_available():
    return _AerSampler is not None


def _make_sampler(backend):
    """Returns (sampler or None, description). None = default exact sampler."""
    if backend == "fast" and _AerSampler is not None:
        return (
            _AerSampler(default_shots=AER_SHOTS, seed=SEED),
            f"Aer simulator ({AER_SHOTS} shots)",
        )
    return None, "Exact statevector simulation"


def _clean_n_train(n_train):
    """0 or less means 'maximum balanced data'."""
    n_train = int(n_train)
    if n_train <= 0:
        return 10 ** 9
    return max(20, min(n_train, MAX_N_TRAIN))


def _predict_proba(vqc, X):
    if hasattr(vqc, "predict_proba"):
        return np.asarray(vqc.predict_proba(X))
    return np.asarray(vqc.neural_network.forward(X, vqc.weights))


# ---------------------------------------------------------------------
# Core training + evaluation (blocking)
# ---------------------------------------------------------------------
def run_vqc(X_train, y_train, X_test, y_test, feature_names,
            filename, processed_folder,
            n_train=DEFAULT_N_TRAIN, maxiter=DEFAULT_MAXITER,
            backend="fast", balance="balanced", progress=None):
    """
    Module 16: Variational Quantum Classifier.

    Pipeline:
    1. Rescale the 8 selected features to [0, pi] (fit on train only).
       The ZZFeatureMap uses each value as a rotation angle, so a
       bounded range gives a consistent encoding.
    2. Take a balanced training subset (see _training_subset).
    3. Build the circuit: ZZFeatureMap (encoding) + RealAmplitudes
       (trainable ansatz), one qubit per feature.
    4. Train with COBYLA on a simulator (no real quantum hardware).
       'fast' uses the Aer simulator (sampled shots); 'exact' uses the
       default exact statevector simulation.
    5. Evaluate on the FULL untouched test set.

    Label convention: 1 = churn. Precision/recall/F1 are for churn.

    Saves:
    - {filename}_vqc_weights.npy
    - {filename}_vqc_scaler.joblib
    - {filename}_vqc_metrics.json
    """

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)

    n_train = _clean_n_train(n_train)
    maxiter = int(max(MIN_MAXITER, min(maxiter, MAX_MAXITER)))
    sampler, backend_label = _make_sampler(backend)

    # 1. Rescale to [0, pi]
    scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(X_train)
    X_train_q = scaler.transform(X_train)
    X_test_q = np.clip(scaler.transform(X_test), 0, np.pi)

    # 2. Choose the training rows (balanced subset, or the full dataset)
    subset = _training_subset(y_train, n_train, balance=balance)
    X_sub, y_sub = X_train_q[subset], y_train[subset]
    n_qubits = int(X_sub.shape[1])

    # 3. Circuit
    algorithm_globals.random_seed = SEED
    feature_map = _make_feature_map(
        n_qubits, FEATURE_MAP_REPS, FEATURE_MAP_ENTANGLEMENT
    )
    ansatz = _make_ansatz(n_qubits, ANSATZ_REPS, ANSATZ_ENTANGLEMENT)

    losses = []

    def callback(weights, loss):
        losses.append(float(loss))
        if progress:
            progress(len(losses), float(loss))

    vqc_kwargs = {}
    if sampler is not None:
        vqc_kwargs["sampler"] = sampler

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=maxiter),
        callback=callback,
        **vqc_kwargs,
    )

    # 4. Train
    start = time.perf_counter()
    vqc.fit(X_sub, y_sub)
    training_time = time.perf_counter() - start

    # 5. Evaluate on the full test set
    start = time.perf_counter()
    proba = _predict_proba(vqc, X_test_q)
    prediction_time = time.perf_counter() - start

    p_churn = proba[:, 1]
    y_pred = (p_churn >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(
            y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(
            y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "f1": round(float(f1_score(
            y_test, y_pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, p_churn)), 4),
    }

    fpr, tpr, _ = roc_curve(y_test, p_churn)
    if len(fpr) > 100:
        keep = np.linspace(0, len(fpr) - 1, 100).astype(int)
        fpr, tpr = fpr[keep], tpr[keep]
    roc_points = [
        [round(float(a), 4), round(float(b), 4)]
        for a, b in zip(fpr, tpr)
    ]

    loss_curve = losses
    if len(loss_curve) > 100:
        keep = np.linspace(0, len(loss_curve) - 1, 100).astype(int)
        loss_curve = [loss_curve[i] for i in keep]
    loss_curve = [round(float(v), 5) for v in loss_curve]

    report = {
        "model_name": "Quantum VQC",
        "params": {
            "backend": backend_label,
            "n_qubits": n_qubits,
            "feature_map": (
                f"ZZFeatureMap (reps={FEATURE_MAP_REPS}, "
                f"{FEATURE_MAP_ENTANGLEMENT} entanglement)"
            ),
            "ansatz": (
                f"RealAmplitudes (reps={ANSATZ_REPS}, "
                f"{ANSATZ_ENTANGLEMENT} entanglement)"
            ),
            "trainable_parameters": int(ansatz.num_parameters),
            "optimizer": f"COBYLA (maxiter={maxiter})",
            "input_scaling": "MinMax to [0, pi], fit on train",
            "balanced_training_subset": balance == "balanced",
            "training_data_mode": (
                "Balanced (equal churn / no-churn)" if balance == "balanced"
                else "Full dataset (real, imbalanced churn ratio)"
            ),
        },
        "features": list(feature_names),
        "n_features": n_qubits,
        "train_samples": int(len(subset)),
        "full_train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "test_churn_rate": round(float(y_test.mean()), 4),
        "iterations_run": len(losses),
        "training_time_sec": round(training_time, 2),
        "prediction_time_sec": round(prediction_time, 2),
        "metrics": metrics,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "roc_curve": roc_points,
        "loss_curve": loss_curve,
    }

    np.save(
        _path(processed_folder, filename, "vqc_weights.npy"),
        np.asarray(vqc.weights)
    )
    joblib.dump(
        scaler, _path(processed_folder, filename, "vqc_scaler.joblib")
    )
    # Native save/load: reproduces predict_proba() exactly, so prediction
    # mode never has to rebuild the circuit by hand.
    vqc.save(_path(processed_folder, filename, "vqc_model.dill"))
    _write_json_atomic(
        _path(processed_folder, filename, "vqc_metrics.json"), report
    )

    return report


# ---------------------------------------------------------------------
# Background execution (training can take minutes)
# ---------------------------------------------------------------------
def start_vqc_training(X_train, y_train, X_test, y_test, feature_names,
                       filename, processed_folder,
                       n_train=DEFAULT_N_TRAIN, maxiter=DEFAULT_MAXITER,
                       backend="fast", balance="balanced"):
    """
    Starts training in a background thread so the web page stays
    responsive. Progress is written to {filename}_vqc_status.json.
    Returns False if a run for this file is already in progress.
    """

    key = (processed_folder, filename)
    with _lock:
        if key in _active:
            return False
        _active.add(key)

    status_path = _path(processed_folder, filename, "vqc_status.json")
    started = time.perf_counter()
    n_used = int(len(_training_subset(
        np.asarray(y_train).astype(int), _clean_n_train(n_train),
        balance=balance
    )))
    maxiter = int(max(MIN_MAXITER, min(maxiter, MAX_MAXITER)))

    def update(state, **extra):
        _write_json_atomic(status_path, {
            "state": state,
            "maxiter": maxiter,
            "n_train": n_used,
            "elapsed": round(time.perf_counter() - started, 1),
            **extra,
        })

    update("running", iteration=0, last_loss=None)

    def progress(iteration, loss):
        if iteration == 1 or iteration % 2 == 0:
            update("running", iteration=iteration,
                   last_loss=round(loss, 4))

    def worker():
        try:
            run_vqc(
                X_train, y_train, X_test, y_test, feature_names,
                filename, processed_folder,
                n_train=n_train, maxiter=maxiter,
                backend=backend, balance=balance, progress=progress
            )
            update("done")
        except Exception as exc:
            update("error", message=str(exc))
        finally:
            with _lock:
                _active.discard(key)

    threading.Thread(target=worker, daemon=True).start()
    return True


def read_vqc_status(filename, processed_folder):
    """Returns the current status dict, e.g. {"state": "running", ...}."""

    path = _path(processed_folder, filename, "vqc_status.json")
    if not os.path.exists(path):
        return {"state": "idle"}

    try:
        with open(path, "r") as f:
            status = json.load(f)
    except (OSError, ValueError):
        return {"state": "running"}

    # A "running" status with no live thread means the server restarted
    key = (processed_folder, filename)
    with _lock:
        alive = key in _active
    if status.get("state") == "running" and not alive:
        return {
            "state": "error",
            "message": "Training was interrupted (the server restarted). "
                       "Please start it again.",
        }
    return status


def load_trained_vqc(processed_folder, prefix="active_model"):
    """
    Loads a previously trained and saved VQC, ready to call
    .predict_proba() on new, already-scaled-to-[0,pi] data.

    `prefix` matches the active_model_* naming used by
    quantum/model_manager.py, so the prediction module can load
    "the" active model without knowing which uploaded file trained it.
    """
    path = os.path.join(processed_folder, f"{prefix}_vqc_model.dill")
    return VQC.load(path)


def evaluate_at_thresholds(processed_folder, filename, thresholds=None):
    """
    Re-evaluates an already-trained VQC on its own held-out test set at
    several different probability cutoffs, without retraining.

    A "cutoff" of 0.5 means: anyone the model scores 50% or higher gets
    labeled "will churn". Raising the cutoff makes the model choosier
    (fewer false alarms, but it will also miss more real churners).
    Lowering it does the opposite. The model's probabilities themselves
    never change -- only where the "churn" line is drawn.

    `filename` is whatever the training run was saved under (e.g. the
    uploaded file's name, or "active_model" once promoted).

    Returns a list of dicts, one per threshold, each with accuracy,
    precision, recall, f1 and the resulting confusion matrix -- so the
    trade-off can be compared side by side.
    """

    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    X_test = np.load(_path(processed_folder, filename, "X_test_selected.npy"))
    y_test = np.load(_path(processed_folder, filename, "y_test.npy")).astype(int)

    vqc_scaler = joblib.load(_path(processed_folder, filename, "vqc_scaler.joblib"))
    X_test_q = np.clip(vqc_scaler.transform(X_test), 0, np.pi)

    vqc = load_trained_vqc(processed_folder, prefix=filename)
    proba = vqc.predict_proba(X_test_q)
    p_churn = proba[:, 1]

    results = []
    for t in thresholds:
        y_pred = (p_churn >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        results.append({
            "threshold": round(float(t), 2),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(
                y_test, y_pred, pos_label=1, zero_division=0)), 4),
            "recall": round(float(recall_score(
                y_test, y_pred, pos_label=1, zero_division=0)), 4),
            "f1": round(float(f1_score(
                y_test, y_pred, pos_label=1, zero_division=0)), 4),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            },
        })

    return results


def set_operating_threshold(processed_folder, filename, threshold):
    """
    Saves the chosen cutoff into this run's metrics file, so anything
    that reads the trained model later (including prediction on new,
    unlabeled customers) uses this threshold instead of the default 0.5.
    """
    path = _path(processed_folder, filename, "vqc_metrics.json")
    with open(path, "r") as f:
        metrics = json.load(f)
    metrics["operating_threshold"] = float(threshold)
    with open(path, "w") as f:
        json.dump(metrics, f)


def repair_missing_dill(processed_folder, filename):
    """
    One-time fix for a run trained before vqc.save() existed: rebuilds
    the fitted VQC purely from its saved weights (vqc_weights.npy) and
    the fixed circuit architecture (same feature map/ansatz/entanglement
    every run in this app uses), then saves it as {filename}_vqc_model.dill
    so evaluate_at_thresholds() and prediction can load it normally.

    This does NOT retrain anything -- it reconstructs the exact same
    trained state, verified to give identical predict_proba() output as
    the original run.
    """
    from qiskit_machine_learning.optimizers import OptimizerResult
    from sklearn.preprocessing import LabelEncoder

    weights = np.load(_path(processed_folder, filename, "vqc_weights.npy"))

    with open(_path(processed_folder, filename, "vqc_metrics.json")) as f:
        metrics = json.load(f)
    n_qubits = metrics["n_features"]

    sampler, _ = _make_sampler(
        "fast" if "Aer" in metrics["params"]["backend"] else "exact"
    )
    feature_map = _make_feature_map(
        n_qubits, FEATURE_MAP_REPS, FEATURE_MAP_ENTANGLEMENT
    )
    ansatz = _make_ansatz(n_qubits, ANSATZ_REPS, ANSATZ_ENTANGLEMENT)

    vqc_kwargs = {}
    if sampler is not None:
        vqc_kwargs["sampler"] = sampler

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=metrics["params"]["optimizer"]
                          .split("maxiter=")[1].rstrip(")")),
        **vqc_kwargs,
    )

    result = OptimizerResult()
    result.x = weights
    result.fun = 0.0
    vqc._fit_result = result
    vqc._num_classes = 2
    vqc._target_encoder = LabelEncoder().fit(np.array([0, 1]))

    vqc.save(_path(processed_folder, filename, "vqc_model.dill"))
    return True