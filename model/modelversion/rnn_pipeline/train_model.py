"""RNN training (Tokenized Sequences + LSTM + Class Weighting)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, models, optimizers

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.rnn_pipeline.data_preprocessing import (  # noqa: E402
    compute_max_len,
    fit_tokenizer,
    load_or_create_splits,
    transform_text,
)
from final_model_pipelines.rnn_pipeline.model_config import (  # noqa: E402
    BATCH_SIZE,
    CLASS_WEIGHT,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EMBEDDING_DIM,
    FEATURE_METHOD,
    IMBALANCE_METHOD,
    LABEL_COL,
    LEARNING_RATE,
    LSTM_UNITS,
    MAX_EPOCHS,
    MAX_VOCAB,
    RANDOM_STATE,
    SAVED_MODEL_DIR,
    TEXT_COL,
)


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_rnn_model(vocab_size: int, max_len: int) -> keras.Model:
    model = models.Sequential(name="FraudRNN")
    model.add(layers.Input(shape=(max_len,)))
    model.add(layers.Embedding(vocab_size, EMBEDDING_DIM, name="embedding"))
    model.add(layers.LSTM(LSTM_UNITS, name="lstm"))
    model.add(layers.Dropout(DROPOUT, name="dropout"))
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

    tokenizer = fit_tokenizer(train_df[TEXT_COL])
    max_len = compute_max_len(tokenizer, train_df[TEXT_COL])
    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

    x_train = transform_text(tokenizer, train_df[TEXT_COL], max_len)
    x_val = transform_text(tokenizer, val_df[TEXT_COL], max_len)
    y_train = train_df[LABEL_COL].values.astype(np.float32)
    y_val = val_df[LABEL_COL].values.astype(np.float32)

    model = build_rnn_model(vocab_size, max_len)
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
    joblib.dump(tokenizer, SAVED_MODEL_DIR / "tokenizer.joblib")

    meta = {
        "feature_method": FEATURE_METHOD,
        "imbalance_method": IMBALANCE_METHOD,
        "class_weight": CLASS_WEIGHT,
        "max_vocab": MAX_VOCAB,
        "max_len": max_len,
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "dropout": DROPOUT,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
    }
    (SAVED_MODEL_DIR / "pipeline_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Model saved: {SAVED_MODEL_DIR}")
    return SAVED_MODEL_DIR


if __name__ == "__main__":
    train()
