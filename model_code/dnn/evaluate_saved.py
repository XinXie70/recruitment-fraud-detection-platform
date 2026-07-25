"""Load saved DNN weights and evaluate once on the fixed test split.

Does not retrain. Threshold comes from model_weights/dnn/training_config.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

_MODEL_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODEL_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_CODE_ROOT))

from dnn.lib import config as cfg
from dnn.lib.data_utils import load_split_csvs, prepare_dataframe
from dnn.lib.features import transform_features
from dnn.lib.metrics_utils import evaluate_binary_predictions, print_evaluation_metrics
from dnn.lib.model import set_global_seeds


def main() -> None:
    set_global_seeds(cfg.RANDOM_STATE)
    weights_dir = cfg.WEIGHTS_DIR
    results_dir = cfg.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    config_path = weights_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}. Download model_weights/dnn first.")

    train_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    threshold = float(train_cfg.get("final_threshold", cfg.DEFAULT_THRESHOLD))

    model = tf.keras.models.load_model(weights_dir / "final_model.keras")
    vectorizer = joblib.load(weights_dir / "tfidf_vectorizer.pkl")
    reducer_path = weights_dir / "dimensionality_reducer.pkl"
    reducer = joblib.load(reducer_path) if reducer_path.exists() else None

    _, _, test_raw = load_split_csvs()
    test_df = prepare_dataframe(test_raw, "test")
    texts = test_df[cfg.TEXT_COLUMN].astype(str).values
    y_true = test_df[cfg.LABEL_COLUMN].astype(int).values

    x_test = transform_features(texts, vectorizer, reducer)
    probs = model.predict(x_test, verbose=0).reshape(-1)
    metrics = evaluate_binary_predictions(y_true, probs, threshold=threshold)
    print_evaluation_metrics("Saved-model test metrics", metrics, threshold=threshold)

    pred_df = pd.DataFrame(
        {
            "record_id": test_df[cfg.RECORD_ID_COLUMN].astype(str).values,
            "true_label": y_true,
            "pred_proba": probs,
            "pred_label": (probs >= threshold).astype(int),
            "threshold": threshold,
            "model_name": "dnn_train_cv",
        }
    )
    out_pred = results_dir / "saved_model_test_predictions.csv"
    out_metrics = results_dir / "saved_model_test_metrics.csv"
    pred_df.to_csv(out_pred, index=False)
    pd.DataFrame([metrics]).to_csv(out_metrics, index=False)
    print(f"Wrote {out_pred}")
    print(f"Wrote {out_metrics}")


if __name__ == "__main__":
    main()
