"""LR pipeline data preprocessing (delegates to shared modules; self-contained training pipeline)."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from final_model_pipelines.data_split import load_or_create_splits
from final_model_pipelines.lr_pipeline.model_config import MAX_FEATURES, NGRAM_RANGE
from final_model_pipelines.text_utils import prepare_text_from_input

__all__ = [
    "load_or_create_splits",
    "fit_tfidf_vectorizer",
    "transform_text",
    "prepare_text_from_input",
]


def fit_tfidf_vectorizer(train_text: pd.Series) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
    )
    vectorizer.fit(train_text)
    return vectorizer


def transform_text(vectorizer: TfidfVectorizer, texts: pd.Series):
    return vectorizer.transform(texts)
