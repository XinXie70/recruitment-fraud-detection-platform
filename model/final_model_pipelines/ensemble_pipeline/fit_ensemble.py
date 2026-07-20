"""Fit member calibrators, soft-voting weights, and ensemble thresholds.

Run from the project root after every member model has been trained:

    python model/final_model_pipelines/ensemble_pipeline/fit_ensemble.py

The script uses only the validation split. The test split remains untouched for
the final evaluation report.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split


PIPELINE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = PIPELINE_DIR.parents[1]
PROJECT_ROOT = MODEL_ROOT.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
for import_path in (PROJECT_ROOT, MODEL_ROOT, BACKEND_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from final_model_pipelines.structured_output import tune_dual_thresholds  # noqa: E402
from services.model_adapter import DEFAULT_MODEL_SPECS, ModelRegistry  # noqa: E402


VALIDATION_PATH = MODEL_ROOT / "final_model_pipelines" / "data" / "splits" / "val.csv"
OUTPUT_PATH = PIPELINE_DIR / "saved_model" / "ensemble_config.json"
RANDOM_STATE = 42


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def _apply_sigmoid(
    probabilities: np.ndarray,
    coefficient: float,
    intercept: float,
) -> np.ndarray:
    values = coefficient * _logit(probabilities) + intercept
    positive = values >= 0
    outputs = np.empty_like(values, dtype=float)
    outputs[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    outputs[~positive] = exp_values / (1 + exp_values)
    return outputs


def _fit_calibrator(
    train_probabilities: np.ndarray,
    train_labels: np.ndarray,
    tune_probabilities: np.ndarray,
    tune_labels: np.ndarray,
) -> tuple[dict, np.ndarray]:
    identity_loss = brier_score_loss(tune_labels, tune_probabilities)
    calibrator = LogisticRegression(random_state=RANDOM_STATE)
    calibrator.fit(_logit(train_probabilities).reshape(-1, 1), train_labels)
    coefficient = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    sigmoid_probabilities = _apply_sigmoid(
        tune_probabilities, coefficient, intercept
    )
    sigmoid_loss = brier_score_loss(tune_labels, sigmoid_probabilities)

    if sigmoid_loss + 1e-8 < identity_loss:
        return (
            {
                "type": "sigmoid",
                "coefficient": coefficient,
                "intercept": intercept,
            },
            sigmoid_probabilities,
        )
    return {"type": "identity"}, tune_probabilities


def _fit_weights(probability_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    model_count = probability_matrix.shape[1]
    initial = np.full(model_count, 1 / model_count)

    def objective(weights: np.ndarray) -> float:
        combined = np.clip(probability_matrix @ weights, 1e-7, 1 - 1e-7)
        return float(log_loss(labels, combined, labels=[0, 1]))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0, 1)] * model_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
    )
    if not result.success:
        raise RuntimeError(f"Weight optimisation failed: {result.message}")
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def fit_ensemble() -> Path:
    frame = pd.read_csv(VALIDATION_PATH)
    texts = frame["combined_text"].fillna("").astype(str).tolist()
    labels = frame["fraudulent"].astype(int).to_numpy()
    model_keys = [spec.key for spec in DEFAULT_MODEL_SPECS]
    registry = ModelRegistry.default()
    raw_outputs = registry.predict_raw_batches(texts, model_keys)

    calibration_indices, tuning_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    calibrations: dict[str, dict] = {}
    tuned_columns: list[np.ndarray] = []
    for key in model_keys:
        probabilities = np.asarray(raw_outputs[key], dtype=float)
        calibration, tuned = _fit_calibrator(
            probabilities[calibration_indices],
            labels[calibration_indices],
            probabilities[tuning_indices],
            labels[tuning_indices],
        )
        calibrations[key] = calibration
        tuned_columns.append(tuned)

    tuned_matrix = np.column_stack(tuned_columns)
    tuning_labels = labels[tuning_indices]
    weights = _fit_weights(tuned_matrix, tuning_labels)
    ensemble_probabilities = tuned_matrix @ weights
    thresholds = tune_dual_thresholds(tuning_labels, ensemble_probabilities)

    if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-8):
        raise RuntimeError("Fitted weights do not sum to one")
    payload = {
        "version": "ensemble-v1-fitted",
        "fitted": True,
        "weight_source": "validation_calibration_split_log_loss_optimisation",
        "low_threshold": thresholds["low_threshold"],
        "high_threshold": thresholds["high_threshold"],
        "models": {
            key: {
                "weight": float(weights[index]),
                "calibration": calibrations[key],
            }
            for index, key in enumerate(model_keys)
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved ensemble configuration to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    fit_ensemble()
