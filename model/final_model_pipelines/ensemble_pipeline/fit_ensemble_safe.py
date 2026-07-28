"""Fit a reproducible eight-model ensemble without touching the test split.

The prediction cache is accepted only when its validation-file SHA-256,
``record_id`` values, and labels exactly match the current fixed validation
split. Rebuild predictions explicitly after any data/model change:

    python model/final_model_pipelines/ensemble_pipeline/fit_ensemble_safe.py \
        --refresh-predictions

Run prediction refresh on the model host with the real local artifacts. The
public remote API applies interactive-input validation and therefore cannot be
used for complete offline evaluation of the fixed split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupShuffleSplit

PIPELINE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = PIPELINE_DIR.parents[1]
PROJECT_ROOT = MODEL_ROOT.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
for import_path in (PROJECT_ROOT, MODEL_ROOT, BACKEND_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

VALIDATION_PATH = PROJECT_ROOT / "data" / "splits" / "validation.csv"
OUTPUT_PATH = PIPELINE_DIR / "saved_model" / "ensemble_config.json"
CACHE_PATH = PIPELINE_DIR / "saved_model" / "_val_predictions.npz"
RANDOM_STATE = 42
REQUIRED_COLUMNS = {"record_id", "combined_text", "label", "group_id"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validation() -> pd.DataFrame:
    frame = pd.read_csv(VALIDATION_PATH)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Validation split is missing columns: {sorted(missing)}")
    if frame["record_id"].duplicated().any():
        raise ValueError("Validation record_id values must be unique")
    labels = set(frame["label"].dropna().astype(int).unique())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError(f"Validation labels must contain binary classes 0 and 1, got {labels}")
    return frame


def _cache_metadata(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "record_ids": frame["record_id"].astype(str).to_numpy(),
        "labels": frame["label"].astype(int).to_numpy(),
        "validation_sha256": np.asarray(_sha256(VALIDATION_PATH)),
    }


def load_verified_cache(frame: pd.DataFrame, model_keys: list[str]) -> dict[str, np.ndarray]:
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(
            f"Prediction cache is missing: {CACHE_PATH}. Run with --refresh-predictions."
        )

    with np.load(CACHE_PATH, allow_pickle=False) as cache:
        required = set(model_keys) | {"record_ids", "labels", "validation_sha256"}
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(
                "Prediction cache has no verifiable provenance "
                f"({sorted(missing)} missing). Run with --refresh-predictions."
            )

        expected = _cache_metadata(frame)
        cached_hash = str(cache["validation_sha256"].item())
        if cached_hash != str(expected["validation_sha256"].item()):
            raise ValueError("Prediction cache was generated from a different validation file")
        if not np.array_equal(cache["record_ids"].astype(str), expected["record_ids"]):
            raise ValueError("Prediction cache record_ids do not match the validation split")
        if not np.array_equal(cache["labels"].astype(int), expected["labels"]):
            raise ValueError("Prediction cache labels do not match the validation split")

        outputs: dict[str, np.ndarray] = {}
        for key in model_keys:
            values = np.asarray(cache[key], dtype=float)
            if values.shape != (len(frame),):
                raise ValueError(f"Cached {key} predictions have shape {values.shape}")
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"Cached {key} predictions are not valid probabilities")
            outputs[key] = values.copy()
        return outputs


def generate_predictions(
    frame: pd.DataFrame,
    model_keys: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    from config import settings
    from services.model_adapter import ModelRegistry

    if settings.model_server_url.strip():
        raise ValueError(
            "Unset MODEL_SERVER_URL and run this command on the model host. "
            "The interactive remote API rejects some fixed evaluation rows."
        )

    texts = frame["combined_text"].fillna("").astype(str).tolist()
    registry = ModelRegistry.default()
    outputs: dict[str, np.ndarray] = {}

    for key in model_keys:
        adapter = registry.adapters[key]
        batches: list[float] = []
        print(f"Predicting {key} ({len(texts)} rows)...", flush=True)
        for start in range(0, len(texts), batch_size):
            batches.extend(adapter.predict_raw_batch(texts[start : start + batch_size]))
        outputs[key] = np.asarray(batches, dtype=float)
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    payload = {**outputs, **_cache_metadata(frame)}
    np.savez_compressed(CACHE_PATH, **payload)
    print(f"Saved verified prediction cache to {CACHE_PATH}")
    return outputs


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def _sigmoid(probabilities: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
    values = coefficient * _logit(probabilities) + intercept
    positive = values >= 0
    outputs = np.empty_like(values)
    outputs[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    outputs[~positive] = exp_values / (1 + exp_values)
    return outputs


def fit_calibrator(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    tuning_probabilities: np.ndarray,
    tuning_labels: np.ndarray,
) -> tuple[dict[str, float | str], np.ndarray]:
    identity_loss = brier_score_loss(tuning_labels, tuning_probabilities)
    calibrator = LogisticRegression(random_state=RANDOM_STATE)
    calibrator.fit(_logit(calibration_probabilities).reshape(-1, 1), calibration_labels)
    coefficient = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    sigmoid_probabilities = _sigmoid(tuning_probabilities, coefficient, intercept)
    sigmoid_loss = brier_score_loss(tuning_labels, sigmoid_probabilities)
    if sigmoid_loss + 1e-8 < identity_loss:
        return {
            "type": "sigmoid",
            "coefficient": coefficient,
            "intercept": intercept,
        }, sigmoid_probabilities
    return {"type": "identity"}, tuning_probabilities


def optimize_weights(probability_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
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


def group_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic group-disjoint 50/50 calibration/tuning split."""
    labels = frame["label"].astype(int).to_numpy()
    target_prevalence = float(labels.mean())
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.5, random_state=RANDOM_STATE)
    for calibration_indices, tuning_indices in splitter.split(
        frame, labels, groups=frame["group_id"]
    ):
        if len(set(labels[calibration_indices])) < 2 or len(set(labels[tuning_indices])) < 2:
            continue
        size_error = abs(len(tuning_indices) / len(frame) - 0.5)
        prevalence_error = abs(float(labels[tuning_indices].mean()) - target_prevalence)
        score = size_error + prevalence_error
        if best is None or score < best[0]:
            best = (score, calibration_indices, tuning_indices)
    if best is None:
        raise RuntimeError("Could not create a group-disjoint split containing both classes")
    return best[1], best[2]


def fit(frame: pd.DataFrame, raw_predictions: dict[str, np.ndarray]) -> dict:
    from final_model_pipelines.structured_output import tune_dual_thresholds

    labels = frame["label"].astype(int).to_numpy()
    calibration_indices, tuning_indices = group_split(frame)
    model_keys = list(raw_predictions)
    calibrations: dict[str, dict[str, float | str]] = {}
    tuned_columns: list[np.ndarray] = []

    for key in model_keys:
        probabilities = raw_predictions[key]
        calibration, tuned = fit_calibrator(
            probabilities[calibration_indices],
            labels[calibration_indices],
            probabilities[tuning_indices],
            labels[tuning_indices],
        )
        calibrations[key] = calibration
        tuned_columns.append(tuned)

    tuned_matrix = np.column_stack(tuned_columns)
    tuning_labels = labels[tuning_indices]
    weights = optimize_weights(tuned_matrix, tuning_labels)
    thresholds = tune_dual_thresholds(tuning_labels, tuned_matrix @ weights)
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-8):
        raise RuntimeError("Fitted weights do not sum to one")

    return {
        "version": "ensemble-v3-group-aware-validation-fitted",
        "fitted": True,
        "weight_source": "group_disjoint_validation_log_loss_optimisation",
        "validation_split_sha256": _sha256(VALIDATION_PATH),
        "random_state": RANDOM_STATE,
        "calibration_rows": len(calibration_indices),
        "tuning_rows": len(tuning_indices),
        "low_threshold": thresholds["low_threshold"],
        "high_threshold": thresholds["high_threshold"],
        "models": {
            key: {"weight": float(weights[index]), "calibration": calibrations[key]}
            for index, key in enumerate(model_keys)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-predictions",
        action="store_true",
        help="Explicitly regenerate all model predictions before fitting.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    from services.model_adapter import DEFAULT_MODEL_SPECS

    frame = load_validation()
    model_keys = [spec.key for spec in DEFAULT_MODEL_SPECS]
    if args.refresh_predictions:
        raw_predictions = generate_predictions(frame, model_keys, args.batch_size)
    else:
        raw_predictions = load_verified_cache(frame, model_keys)

    payload = fit(frame, raw_predictions)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved fitted ensemble configuration to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
