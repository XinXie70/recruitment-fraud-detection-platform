"""DNN training (BoW + Class Weighting + Keras)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import callbacks, layers, models, optimizers

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.dnn_pipeline.data_preprocessing import (  # noqa: E402
    fit_bow_vectorizer,
    load_or_create_splits,
    transform_text,
)
from final_model_pipelines.dnn_pipeline.model_config import (  # noqa: E402
    BATCH_SIZE,
    CLASS_WEIGHT,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    FEATURE_METHOD,
    HIDDEN_DIMS,
    IMBALANCE_METHOD,
    LABEL_COL,
    LEARNING_RATE,
    MAX_EPOCHS,
    RANDOM_STATE,
    SAVED_MODEL_DIR,
    TEXT_COL,
)


def _to_dense(x) -> np.ndarray:
    if issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_dnn_model(input_dim: int) -> keras.Model:
    model = models.Sequential(name="FraudDNN")
    model.add(layers.Input(shape=(input_dim,)))
    for i, units in enumerate(HIDDEN_DIMS):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{i + 1}"))
        model.add(layers.Dropout(DROPOUT, name=f"dropout_{i + 1}"))
    model.add(layers.Dense(1, activation="sigmoid", name="output"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train() -> Path:
    _set_seeds(RANDOM_STATE)
    train_df, val_df, _ = load_or_create_splits()

    vectorizer = fit_bow_vectorizer(train_df[TEXT_COL])
    x_train = _to_dense(transform_text(vectorizer, train_df[TEXT_COL]))
    x_val = _to_dense(transform_text(vectorizer, val_df[TEXT_COL]))
    y_train = train_df[LABEL_COL].values.astype(np.float32)
    y_val = val_df[LABEL_COL].values.astype(np.float32)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)

    model = build_dnn_model(x_train.shape[1])
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    fit_class_weight = {int(k): float(v) for k, v in CLASS_WEIGHT.items()}
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=fit_class_weight,
        callbacks=[early_stop],
        verbose=1,
    )

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(SAVED_MODEL_DIR / "model.keras")
    joblib.dump(vectorizer, SAVED_MODEL_DIR / "vectorizer.joblib")
    joblib.dump(scaler, SAVED_MODEL_DIR / "scaler.joblib")

    meta = {
        "feature_method": FEATURE_METHOD,
        "imbalance_method": IMBALANCE_METHOD,
        "class_weight": CLASS_WEIGHT,
        "hidden_dims": list(HIDDEN_DIMS),
        "dropout": DROPOUT,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
    }
    (SAVED_MODEL_DIR / "pipeline_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Model saved: {SAVED_MODEL_DIR}")
    return SAVED_MODEL_DIR


if __name__ == "__main__":
    train()
