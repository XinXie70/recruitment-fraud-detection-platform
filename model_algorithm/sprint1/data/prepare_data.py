"""Sprint1 data preparation for EMSCAD.

Steps:
  1) Load emscad_v1.csv
  2) Clean / normalise the five text fields; missing text -> empty string
  3) Drop exact-duplicate combined_text rows (keep first by source order)
  4) Stratified 70 / 15 / 15 train / validation / test split (seed=42)

Outputs under ./splits/:
  train.csv, validation.csv, test.csv, split_meta.json
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent
RAW_CSV = DATA_DIR / "emscad_v1.csv"
OUT_DIR = DATA_DIR / "splits"
SEED = 42

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]


def clean_text(value) -> str:
    """Shared light text normalisation (DATA_CONTRACT_V1)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def convert_label(value) -> int:
    v = str(value).strip().lower()
    if v in {"0", "f", "false"}:
        return 0
    if v in {"1", "t", "true"}:
        return 1
    raise ValueError(f"Unexpected fraudulent label: {value!r}")


def combine_fields(row: pd.Series) -> str:
    parts = [clean_text(row.get(col, "")) for col in TEXT_COLUMNS]
    return "\n".join(part for part in parts if part)


def build_frame(raw_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    need = TEXT_COLUMNS + ["fraudulent"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw CSV missing columns: {missing}")

    rows = []
    for i, row in raw.iterrows():
        rows.append(
            {
                "record_id": f"emscad_{int(i) + 1:05d}",
                "combined_text": combine_fields(row),
                "label": convert_label(row["fraudulent"]),
            }
        )
    return pd.DataFrame(rows)


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    # Keep first occurrence in original source order.
    out = df.drop_duplicates(subset=["combined_text"], keep="first").reset_index(drop=True)
    return out, before - len(out)


def stratified_split(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """70% train, 15% validation, 15% test with stratification on label."""
    idx = np.arange(len(df))
    labels = df["label"]

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        stratify=labels,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=labels.iloc[temp_idx],
        random_state=seed,
    )
    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "validation": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw dataset: {RAW_CSV}")

    print(f"Loading: {RAW_CSV}")
    df = build_frame(RAW_CSV)
    print(f"Raw rows: {len(df)} | fraud: {int((df['label'] == 1).sum())}")

    empty_n = int((df["combined_text"].astype(str).str.strip() == "").sum())
    print(f"Empty combined_text after missing-text handling: {empty_n}")

    df, n_dropped = drop_exact_duplicates(df)
    print(f"Dropped exact duplicates: {n_dropped}")
    print(f"After dedup: {len(df)} | fraud: {int((df['label'] == 1).sum())}")

    splits = stratified_split(df, seed=SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    counts = {}
    for name, part in splits.items():
        path = OUT_DIR / f"{name}.csv"
        part.to_csv(path, index=False)
        counts[name] = {
            "n": int(len(part)),
            "fraud": int((part["label"] == 1).sum()),
            "fraud_rate": float((part["label"] == 1).mean()),
        }
        print(f"Wrote {path.name}: n={counts[name]['n']} fraud={counts[name]['fraud']}")

    meta = {
        "seed": SEED,
        "protocol": "70/15/15 stratified",
        "source": str(RAW_CSV.name),
        "exact_duplicates_removed": int(n_dropped),
        "empty_combined_text_rows_kept": empty_n,
        "split_counts": counts,
        "n_total_after_dedup": int(len(df)),
        "columns": ["record_id", "combined_text", "label"],
        "cleaning": [
            "missing text -> empty string",
            "HTML entity decode",
            "HTML tag removal",
            "Unicode NFKC",
            "newline / whitespace normalisation",
            "exact duplicate combined_text dropped (keep first)",
        ],
    }
    (OUT_DIR / "split_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
