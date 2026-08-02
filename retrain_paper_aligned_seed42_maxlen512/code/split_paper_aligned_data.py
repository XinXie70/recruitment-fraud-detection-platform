#!/usr/bin/env python
"""Paper-aligned data preparation + split (seed=42).

Protocol (matches project Paper-aligned arm):
  1) Load EMSCAD raw CSV (18 columns)
  2) Build Condition-A style rows: record_id + combined_text(5 fields) + label
  3) Build Paper-aligned model_text (structured tags + section word caps)
  4) Stratified 80/20 test, then 10% of train_pool -> validation (seed=42)

Usage (from this bundle root):
  python code/split_paper_aligned_data.py
  python code/split_paper_aligned_data.py --raw data/raw/emscad_v1.csv --out data/splits
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
ID_COLUMN = "record_id"
LABEL_COLUMN = "label"

TEXT_COLUMNS_5 = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

SECTION_WORD_CAPS = {
    "title": 20,
    "company_profile": 80,
    "description": 300,
    "requirements": 120,
    "benefits": 80,
}


def clean_text_light(value) -> str:
    """Project shared light cleaning (Condition A / prepare_data)."""
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


def combine_five_fields(row: pd.Series) -> str:
    parts = [clean_text_light(row.get(c, "")) for c in TEXT_COLUMNS_5]
    return "\n".join(p for p in parts if p)


def _cap_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def build_structured_combined_text(combined: str) -> str:
    """Paper-aligned structured text (same as run_bert_paper_vs_project)."""
    if combined is None or (isinstance(combined, float) and pd.isna(combined)):
        return ""
    lines = [ln.strip() for ln in str(combined).split("\n") if ln.strip()]
    if not lines:
        return ""

    parts: List[str] = []
    if len(lines) == 1:
        parts.append(f"[TITLE] {_cap_words(lines[0], SECTION_WORD_CAPS['title'])}")
        return "\n".join(parts)

    parts.append(f"[TITLE] {_cap_words(lines[0], SECTION_WORD_CAPS['title'])}")
    rest = lines[1:]
    if len(rest) == 1:
        parts.append(
            f"[DESCRIPTION] {_cap_words(rest[0], SECTION_WORD_CAPS['description'])}"
        )
        return "\n".join(parts)

    parts.append(
        f"[COMPANY PROFILE] {_cap_words(rest[0], SECTION_WORD_CAPS['company_profile'])}"
    )
    body = rest[1:]
    if not body:
        return "\n".join(parts)

    n = len(body)
    if n == 1:
        parts.append(
            f"[DESCRIPTION] {_cap_words(body[0], SECTION_WORD_CAPS['description'])}"
        )
        return "\n".join(parts)

    c1 = max(1, n // 3)
    c2 = max(c1 + 1, (2 * n) // 3)
    desc = " ".join(body[:c1])
    reqs = " ".join(body[c1:c2])
    bens = " ".join(body[c2:])
    parts.append(f"[DESCRIPTION] {_cap_words(desc, SECTION_WORD_CAPS['description'])}")
    if reqs.strip():
        parts.append(
            f"[REQUIREMENTS] {_cap_words(reqs, SECTION_WORD_CAPS['requirements'])}"
        )
    if bens.strip():
        parts.append(f"[BENEFITS] {_cap_words(bens, SECTION_WORD_CAPS['benefits'])}")
    return "\n".join(parts)


def load_raw_to_frame(raw_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    need = TEXT_COLUMNS_5 + ["fraudulent"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw CSV missing columns: {missing}")

    rows = []
    for i, row in raw.iterrows():
        rid = f"emscad_{i + 1:05d}"
        combined = combine_five_fields(row)
        model_text = build_structured_combined_text(combined)
        if not str(model_text).strip():
            model_text = combined
        rows.append(
            {
                ID_COLUMN: rid,
                "combined_text": combined,
                "model_text": model_text,
                LABEL_COLUMN: convert_label(row["fraudulent"]),
                "title": str(model_text).split("\n", 1)[0][:120],
                "company": "",
            }
        )
    return pd.DataFrame(rows)


def make_assignment(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """80/20 stratified, then 10% of train_pool -> validation."""
    labels = df[LABEL_COLUMN]
    idx = np.arange(len(df))
    train_pool, test_idx = train_test_split(
        idx, test_size=0.20, stratify=labels, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_pool,
        test_size=0.10,
        stratify=labels.iloc[train_pool],
        random_state=seed,
    )
    split = pd.Series("", index=df.index)
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "validation"
    split.iloc[test_idx] = "test"
    return pd.DataFrame(
        {
            ID_COLUMN: df[ID_COLUMN].astype(str),
            "seed": seed,
            "split": split.astype(str),
        }
    )


def write_splits(df: pd.DataFrame, assign: pd.DataFrame, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    assign.to_csv(out_dir / "assignments.csv.gz", index=False, compression="gzip")
    merged = assign.merge(df, on=ID_COLUMN, how="left", validate="one_to_one")
    counts = {}
    for name in ("train", "validation", "test"):
        part = merged[merged["split"] == name].copy().reset_index(drop=True)
        part.to_csv(out_dir / f"{name}.csv.gz", index=False, compression="gzip")
        counts[name] = {
            "n": int(len(part)),
            "fraud": int((part[LABEL_COLUMN] == 1).sum()),
        }
    meta = {
        "seed": int(assign["seed"].iloc[0]),
        "protocol": "80/20 + 10% val-of-train",
        "split_counts": counts,
        "n_total": int(len(df)),
    }
    (out_dir / "split_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw",
        type=Path,
        default=BUNDLE_ROOT / "data" / "raw" / "emscad_v1.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=BUNDLE_ROOT / "data" / "splits",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--also_save_processed",
        action="store_true",
        help="Also write data/processed/emscad_condition_a_from_raw.csv.gz",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw.exists():
        raise FileNotFoundError(
            f"Raw EMSCAD not found: {args.raw}\n"
            "Place emscad_v1.csv under data/raw/ first."
        )
    print(f"Loading raw: {args.raw}")
    df = load_raw_to_frame(args.raw)
    print(f"Built frame n={len(df)} fraud={int((df[LABEL_COLUMN]==1).sum())}")

    if args.also_save_processed:
        proc_dir = BUNDLE_ROOT / "data" / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        proc_path = proc_dir / "emscad_condition_a_from_raw.csv.gz"
        df[[ID_COLUMN, "combined_text", LABEL_COLUMN]].to_csv(
            proc_path, index=False, compression="gzip"
        )
        print(f"Wrote processed: {proc_path}")

    assign = make_assignment(df, seed=args.seed)
    meta = write_splits(df, assign, args.out)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Splits written to: {args.out}")


if __name__ == "__main__":
    main()
