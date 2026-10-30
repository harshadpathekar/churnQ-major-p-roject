"""
One-off repair script.

Run this once from your project root (same folder as app.py):
    python repair_and_activate.py

It will:
1. Rebuild Telco dataset-35.csv_vqc_model.dill from the saved weights
   (no retraining -- same fix as before, just applied to this run).
2. Re-activate that training run, now that model_manager.py includes
   vqc_model.dill in what gets promoted to active_model_*.

Safe to delete after it runs successfully.
"""

import os
import json
import numpy as np

import config
from quantum.model_manager import activate_model, ARTIFACT_SUFFIXES

TRAINED_FROM_FILE = "Telco dataset-35.csv"
PROCESSED_FOLDER = config.PROCESSED_FOLDER


def _path(suffix):
    return os.path.join(PROCESSED_FOLDER, f"{TRAINED_FROM_FILE}_{suffix}")


def repair_missing_dill():
    """Same logic as quantum_vqc.repair_missing_dill(), with the
    maxiter string->int bug fixed."""
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.optimizers import OptimizerResult
    from sklearn.preprocessing import LabelEncoder

    # Reuse the real helpers so the circuit architecture is identical
    # to what training used.
    from quantum.quantum_vqc import (
        _make_feature_map, _make_ansatz, _make_sampler,
        FEATURE_MAP_REPS, FEATURE_MAP_ENTANGLEMENT,
        ANSATZ_REPS, ANSATZ_ENTANGLEMENT,
    )
    try:
        from qiskit_machine_learning.optimizers import COBYLA
    except ImportError:
        from qiskit_algorithms.optimizers import COBYLA

    weights = np.load(_path("vqc_weights.npy"))

    with open(_path("vqc_metrics.json")) as f:
        metrics = json.load(f)
    n_qubits = metrics["n_features"]

    sampler, _ = _make_sampler(
        "fast" if "Aer" in metrics["params"]["backend"] else "exact"
    )
    feature_map = _make_feature_map(n_qubits, FEATURE_MAP_REPS, FEATURE_MAP_ENTANGLEMENT)
    ansatz = _make_ansatz(n_qubits, ANSATZ_REPS, ANSATZ_ENTANGLEMENT)

    # FIX: wrap in int() -- the original line left this as a string,
    # which would have raised or misbehaved inside COBYLA.
    maxiter_str = metrics["params"]["optimizer"].split("maxiter=")[1].rstrip(")")
    maxiter = int(maxiter_str)

    vqc_kwargs = {}
    if sampler is not None:
        vqc_kwargs["sampler"] = sampler

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=maxiter),
        **vqc_kwargs,
    )

    result = OptimizerResult()
    result.x = weights
    result.fun = 0.0
    vqc._fit_result = result
    vqc._num_classes = 2
    vqc._target_encoder = LabelEncoder().fit(np.array([0, 1]))

    vqc.save(_path("vqc_model.dill"))
    print(f"Wrote {_path('vqc_model.dill')}")


def verify_artifacts_present():
    missing = [s for s in ARTIFACT_SUFFIXES if not os.path.exists(_path(s))]
    if missing:
        print(f"WARNING: still missing before activation: {missing}")
        return False
    return True


if __name__ == "__main__":
    print(f"Repairing: {TRAINED_FROM_FILE}")
    print(f"Processed folder: {PROCESSED_FOLDER}")

    if os.path.exists(_path("vqc_model.dill")):
        print("vqc_model.dill already exists -- skipping repair step.")
    else:
        repair_missing_dill()

    if not verify_artifacts_present():
        raise SystemExit(
            "Cannot activate: one or more required files are still missing. "
            "Check the WARNING above."
        )

    ok, missing_suffix = activate_model(TRAINED_FROM_FILE, PROCESSED_FOLDER)
    if ok:
        print("Activated successfully. active_model_* files now include vqc_model.dill.")
    else:
        print(f"Activation failed -- missing: {missing_suffix}")
