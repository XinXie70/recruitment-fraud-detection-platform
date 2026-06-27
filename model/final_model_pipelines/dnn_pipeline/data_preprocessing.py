"""DNN pipeline data preprocessing (delegates to shared modules; self-contained training pipeline)."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from final_model_pipelines.data_split import load_or_create_splits
from final_model_pipelines.dnn_pipeline.model_config import MAX_FEATURES, NGRAM_RANGE
from final_model_pipelines.text_utils import prepare_text_from_input

__all__ = [
    "load_or_create_splits",
    "fit_bow_vectorizer",
    "transform_text",
    "prepare_text_from_input",
]


def fit_bow_vectorizer(train_text: pd.Series) -> CountVectorizer:
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
    )
    vectorizer.fit(train_text)
    return vectorizer


def transform_text(vectorizer: CountVectorizer, texts: pd.Series):
    return vectorizer.transform(texts)
