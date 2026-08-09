"""Text cleaning and combined_text construction (shared by LR / DNN)."""

from __future__ import annotations

import re

import pandas as pd

from final_model_pipelines.shared_config import LABEL_COL, TEXT_COL, TEXT_FIELDS


def clean_html(text: str) -> str:
    if pd.isna(text) or text == "":
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def build_combined_text(row: pd.Series) -> str:
    parts = [clean_html(row.get(field, "")) for field in TEXT_FIELDS]
    return " ".join(part for part in parts if part)


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Boolean labels compare equal to 1/0, so the numeric keys cover both forms.
    out[LABEL_COL] = out[LABEL_COL].map({"t": 1, "f": 0, 1: 1, 0: 0})
    if out[LABEL_COL].isna().any():
        raise ValueError("Unrecognized label values in the fraudulent column.")
    return out


def preprocess_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Full text preprocessing: HTML cleaning + merge five fields into combined_text."""
    df = encode_labels(df)
    for field in TEXT_FIELDS:
        if field in df.columns:
            df[field] = df[field].apply(clean_html)
    df[TEXT_COL] = df.apply(build_combined_text, axis=1)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


def prepare_text_from_input(input_text: str) -> str:
    """Apply the same cleaning as training for a single prediction input."""
    return clean_html(input_text)
