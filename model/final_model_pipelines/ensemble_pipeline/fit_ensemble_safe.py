"""Memory-safe ensemble fitting: processes one model at a time."""

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
for p in (PROJECT_ROOT, MODEL_ROOT, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

VALIDATION_PATH = MODEL_ROOT / "modelversion" / "data" / "splits" / "val.csv"
OUTPUT_PATH = PIPELINE_DIR / "saved_model" / "ensemble_config.json"
CACHE_PATH = PIPELINE_DIR / "saved_model" / "_val_predictions.npz"
RANDOM_STATE = 42


def _logit(probs):
    return np.log(np.clip(probs, 1e-7, 1 - 1e-7))


def _sigmoid(probs, a, b):
    v = a * _logit(probs) + b
    pos = v >= 0
    out = np.empty_like(v)
    out[pos] = 1 / (1 + np.exp(-v[pos]))
    out[~pos] = np.exp(v[~pos]) / (1 + np.exp(v[~pos]))
    return out


def fit_calibrator(train_p, train_y, tune_p, tune_y):
    id_loss = brier_score_loss(tune_y, tune_p)
    cal = LogisticRegression(random_state=RANDOM_STATE)
    cal.fit(_logit(train_p).reshape(-1, 1), train_y)
    a, b = float(cal.coef_[0, 0]), float(cal.intercept_[0])
    sig_p = _sigmoid(tune_p, a, b)
    sig_loss = brier_score_loss(tune_y, sig_p)
    if sig_loss + 1e-8 < id_loss:
        return {"type": "sigmoid", "coefficient": a, "intercept": b}, sig_p
    return {"type": "identity"}, tune_p


def optimize_weights(pmat, labels):
    n = pmat.shape[1]
    init = np.full(n, 1 / n)

    def objective(w):
        combined = np.clip(pmat @ w, 1e-7, 1 - 1e-7)
        return float(log_loss(labels, combined, labels=[0, 1]))

    res = minimize(
        objective, init, method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    if not res.success:
        raise RuntimeError(f"Weight opt failed: {res.message}")
    w = np.clip(res.x, 0, 1)
    return w / w.sum()


def tune_thresholds(labels, probs):
    from final_model_pipelines.structured_output import tune_dual_thresholds
    return tune_dual_thresholds(labels, probs)


def predict_one_model(key, texts):
    from services.model_adapter import DEFAULT_MODEL_SPECS, ModelAdapter
    import gc
    spec = next(s for s in DEFAULT_MODEL_SPECS if s.key == key)
    adapter = ModelAdapter(spec)
    scores = adapter.predict_raw_batch(texts)
    del adapter
    gc.collect()
    return np.array(scores, dtype=float)


def main():
    print("Loading validation data...")
    df = pd.read_csv(VALIDATION_PATH)
    texts = df["combined_text"].fillna("").astype(str).tolist()
    labels = df["fraudulent"].astype(int).to_numpy()
    print(f"  {len(texts)} samples, {labels.sum()} fraudulent")

    from services.model_adapter import DEFAULT_MODEL_SPECS
    model_keys = [s.key for s in DEFAULT_MODEL_SPECS]
    print(f"Models: {model_keys}")

    raw_predictions = {}
    for key in model_keys:
        print(f"  Predicting {key}...", end=" ", flush=True)
        raw_predictions[key] = predict_one_model(key, texts)
        print(f"mean={raw_predictions[key].mean():.4f}")

    np.savez_compressed(CACHE_PATH, **raw_predictions, labels=labels)
    print(f"Cached to {CACHE_PATH}")

    calib_idx, tune_idx = train_test_split(
        np.arange(len(labels)), test_size=0.5,
        random_state=RANDOM_STATE, stratify=labels,
    )
    calibrations = {}
    tuned_cols = []
    for key in model_keys:
        probs = raw_predictions[key]
        cal, tuned = fit_calibrator(
            probs[calib_idx], labels[calib_idx],
            probs[tune_idx], labels[tune_idx],
        )
        calibrations[key] = cal
        tuned_cols.append(tuned)
        print(f"  {key}: {cal['type']}", end="")
        if cal['type'] == 'sigmoid':
            print(f" (a={cal['coefficient']:.3f}, b={cal['intercept']:.3f})")
        else:
            print()

    pmat = np.column_stack(tuned_cols)
    tune_labels = labels[tune_idx]
    weights = optimize_weights(pmat, tune_labels)
    ensemble_probs = pmat @ weights

    thresholds = tune_thresholds(tune_labels, ensemble_probs)

    assert math.isclose(float(weights.sum()), 1.0, abs_tol=1e-8)
    payload = {
        "version": "ensemble-v1-fitted",
        "fitted": True,
        "weight_source": "validation_log_loss_optimisation_sigmoid_calibrated",
        "low_threshold": thresholds["low_threshold"],
        "high_threshold": thresholds["high_threshold"],
        "models": {
            key: {
                "weight": float(weights[i]),
                "calibration": calibrations[key],
            }
            for i, key in enumerate(model_keys)
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== ensemble_config.json ===")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
