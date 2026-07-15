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
    out[LABEL_COL] = out[LABEL_COL].map({"t": 1, "f": 0, True: 1, False: 0, 1: 1, 0: 0})
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


def normalize_combined_text_key(text: str) -> str:
    """Normalize combined_text for exact-duplicate matching across rows."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _nonempty_text_field_count(df: pd.DataFrame) -> pd.Series:
    richness = pd.Series(0, index=df.index, dtype=int)
    for field in TEXT_FIELDS:
        if field in df.columns:
            richness = richness + df[field].fillna("").astype(str).str.strip().ne("").astype(int)
    return richness


def deduplicate_by_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with identical normalized combined_text.

    When duplicates exist, keep the row with the most non-empty text fields
    (title / company_profile / description / requirements / benefits). Ties keep
    the earlier row in the current frame order.
    """
    if TEXT_COL not in df.columns:
        raise ValueError(f"Expected column {TEXT_COL!r} before deduplication.")

    work = df.copy()
    before = len(work)
    work["_dedup_key"] = work[TEXT_COL].map(normalize_combined_text_key)
    work["_richness"] = _nonempty_text_field_count(work)
    work = work.sort_values("_richness", ascending=False, kind="mergesort")
    work = work.drop_duplicates(subset=["_dedup_key"], keep="first")
    work = work.sort_index()
    work = work.drop(columns=["_dedup_key", "_richness"])
    removed = before - len(work)
    if removed:
        print(
            f"Deduplicated combined_text: removed {removed} duplicate rows "
            f"({before} → {len(work)})"
        )
    return work.reset_index(drop=True)


def prepare_text_from_input(input_text: str) -> str:
    """Apply the same cleaning as training for a single prediction input."""
    return clean_html(input_text)
