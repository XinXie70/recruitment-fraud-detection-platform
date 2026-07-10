"""Bi-LSTM pipeline data preprocessing (delegates to shared modules; self-contained training pipeline)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from final_model_pipelines.data_split import load_or_create_splits
from final_model_pipelines.bilstm_pipeline.model_config import MAX_LEN, MAX_LEN_PERCENTILE, MAX_VOCAB
from final_model_pipelines.text_utils import prepare_text_from_input

__all__ = [
    "load_or_create_splits",
    "fit_tokenizer",
    "compute_max_len",
    "transform_text",
    "prepare_text_from_input",
    "load_sequence_config",
]


def fit_tokenizer(train_text: pd.Series) -> Tokenizer:
    tokenizer = Tokenizer(num_words=MAX_VOCAB, lower=True, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text.astype(str).tolist())
    return tokenizer


def compute_max_len(tokenizer: Tokenizer, train_text: pd.Series, percentile: float = MAX_LEN_PERCENTILE) -> int:
    sequences = tokenizer.texts_to_sequences(train_text.astype(str).tolist())
    lengths = [len(seq) for seq in sequences if seq]
    if not lengths:
        return MAX_LEN
    return min(MAX_LEN, int(np.percentile(lengths, percentile)))


def transform_text(tokenizer: Tokenizer, texts: pd.Series, max_len: int) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts.astype(str).tolist())
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")


def load_sequence_config(saved_model_dir: Path) -> dict:
    config_path = saved_model_dir / "pipeline_config.json"
    if not config_path.exists():
        return {"max_len": MAX_LEN}
    return json.loads(config_path.read_text(encoding="utf-8"))
