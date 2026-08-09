"""Dataset loading, quality checks, and PyTorch Dataset wrappers.

Strictly uses the pre-made train / validation / test CSV files.
Never re-splits or merges them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from config import (
    GROUP_COLUMN,
    ID_COLUMN,
    LABEL_COLUMN,
    TEST_CSV,
    TRAIN_CSV,
    VALIDATION_CSV,
)
from preprocessing import (
    derive_company_preview,
    derive_title_preview,
    prepare_dataframe_text,
)


SPLIT_FILES = {
    "train": TRAIN_CSV,
    "validation": VALIDATION_CSV,
    "test": TEST_CSV,
}


def load_split(split: str, path: Optional[Path] = None) -> pd.DataFrame:
    """Load one fixed split CSV. Does not merge or re-partition."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_FILES)}")
    csv_path = Path(path) if path is not None else SPLIT_FILES[split]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Required split file missing: {csv_path}. "
            "Do not create a new split; use the project train/validation/test CSVs."
        )
    df = pd.read_csv(csv_path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{csv_path} missing required column '{LABEL_COLUMN}'")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    # Paper-aligned splits already store the exact model_text used in training.
    # Do not rebuild it — retagging would change the input and break reproduction.
    if "model_text" not in df.columns or df["model_text"].isna().all():
        df = prepare_dataframe_text(df)
    else:
        df["model_text"] = df["model_text"].fillna("").astype(str)
    df["split"] = split
    df["row_index"] = np.arange(len(df), dtype=np.int64)
    if ID_COLUMN not in df.columns:
        df[ID_COLUMN] = [f"{split}_{i:05d}" for i in range(len(df))]
    if "title" not in df.columns:
        df["title"] = df["model_text"].map(derive_title_preview)
    if "company" not in df.columns:
        # Splits do not store company separately; derive a preview for analysis.
        df["company"] = df["model_text"].map(derive_company_preview)
    return df


def load_all_splits(
    train_path: Optional[Path] = None,
    validation_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Load the three fixed splits independently (no merging)."""
    return {
        "train": load_split("train", train_path),
        "validation": load_split("validation", validation_path),
        "test": load_split("test", test_path),
    }


def _label_stats(df: pd.DataFrame) -> Dict[str, Any]:
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    n0 = int(counts.get(0, 0))
    n1 = int(counts.get(1, 0))
    total = len(df)
    return {
        "n_samples": total,
        "n_legitimate": n0,
        "n_fraudulent": n1,
        "class_ratio_legitimate": float(n0 / total) if total else 0.0,
        "class_ratio_fraudulent": float(n1 / total) if total else 0.0,
        "label_counts": {str(k): int(v) for k, v in sorted(counts.items())},
    }


def _text_stats(df: pd.DataFrame) -> Dict[str, Any]:
    texts = df["model_text"].fillna("")
    lengths = texts.str.len()
    return {
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "empty_text_count": int((texts.str.strip() == "").sum()),
        "exact_duplicate_text_count": int(texts.duplicated().sum()),
        "duplicate_id_count": int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df.columns else None,
        "text_length": {
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "median": float(lengths.median()) if len(lengths) else 0.0,
            "min": int(lengths.min()) if len(lengths) else 0,
            "max": int(lengths.max()) if len(lengths) else 0,
            "p90": float(lengths.quantile(0.90)) if len(lengths) else 0.0,
            "p95": float(lengths.quantile(0.95)) if len(lengths) else 0.0,
            "p99": float(lengths.quantile(0.99)) if len(lengths) else 0.0,
        },
        "columns": list(df.columns),
        "title_unique": int(df["title"].nunique()) if "title" in df.columns else None,
        "company_unique": int(df["company"].nunique()) if "company" in df.columns else None,
    }


def compute_token_length_stats(
    texts: Sequence[str],
    tokenizer,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Tokenise without truncation to estimate length distribution."""
    sample = list(texts)
    if max_samples is not None and len(sample) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(sample), size=max_samples, replace=False)
        sample = [sample[i] for i in idx]

    lengths: List[int] = []
    for text in sample:
        enc = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        lengths.append(len(enc["input_ids"]))

    arr = np.asarray(lengths, dtype=np.float64)

    def pct_over(limit: int) -> float:
        return float((arr > limit).mean()) if len(arr) else 0.0

    p_over_256 = pct_over(256)
    p_over_384 = pct_over(384)
    p_over_512 = pct_over(512)
    reason = (
        f"Measured train token lengths: median≈{float(np.median(arr)) if len(arr) else 0:.0f}, "
        f"{p_over_256:.1%} >256, {p_over_384:.1%} >384, {p_over_512:.1%} >512. "
        "Default max_length=384 is chosen for RTX 3060 with batch=4 / accum=4; "
        "try 512 only after an OOM check. Long ads may still be truncated."
    )
    return {
        "n_texts_measured": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
        "pct_over_128": pct_over(128),
        "pct_over_256": p_over_256,
        "pct_over_384": p_over_384,
        "pct_over_512": p_over_512,
        "recommended_max_length": 384,
        "recommendation_reason": reason,
    }


def build_data_quality_report(
    splits: Dict[str, pd.DataFrame],
    token_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce a report only — never re-split based on findings."""
    per_split = {}
    for name, df in splits.items():
        per_split[name] = {**_label_stats(df), **_text_stats(df)}

    train, val, test = splits["train"], splits["validation"], splits["test"]

    def set_overlap(a: pd.Series, b: pd.Series) -> int:
        return int(len(set(a.astype(str)) & set(b.astype(str))))

    cross = {
        "identical_text_train_validation": set_overlap(train["model_text"], val["model_text"]),
        "identical_text_train_test": set_overlap(train["model_text"], test["model_text"]),
        "identical_text_validation_test": set_overlap(val["model_text"], test["model_text"]),
        "identical_id_train_validation": set_overlap(train[ID_COLUMN], val[ID_COLUMN]),
        "identical_id_train_test": set_overlap(train[ID_COLUMN], test[ID_COLUMN]),
        "identical_id_validation_test": set_overlap(val[ID_COLUMN], test[ID_COLUMN]),
        "note": (
            "Cross-split overlaps are reported for awareness only. "
            "This module never reassigns rows across splits."
        ),
    }

    if GROUP_COLUMN in train.columns:
        cross.update(
            {
                "group_overlap_train_validation": set_overlap(train[GROUP_COLUMN], val[GROUP_COLUMN]),
                "group_overlap_train_test": set_overlap(train[GROUP_COLUMN], test[GROUP_COLUMN]),
                "group_overlap_validation_test": set_overlap(val[GROUP_COLUMN], test[GROUP_COLUMN]),
            }
        )

    # Company+title combo check when both derived/present.
    if "company" in train.columns and "title" in train.columns:
        def combo(df: pd.DataFrame) -> pd.Series:
            return df["company"].astype(str) + " || " + df["title"].astype(str)

        cross["company_title_overlap_train_validation"] = set_overlap(combo(train), combo(val))
        cross["company_title_overlap_train_test"] = set_overlap(combo(train), combo(test))
        cross["company_title_overlap_validation_test"] = set_overlap(combo(val), combo(test))

    report: Dict[str, Any] = {
        "split_files": {k: str(SPLIT_FILES[k]) for k in SPLIT_FILES},
        "per_split": per_split,
        "cross_split_checks": cross,
        "policy": {
            "no_resplit": True,
            "no_merge": True,
            "no_cross_validation": True,
            "oversample_train_only": True,
        },
    }
    if token_stats is not None:
        report["token_length_stats_train"] = token_stats
    return report


def compute_class_weights(
    y: np.ndarray,
    n_classes: int = 2,
    *,
    transform: str = "none",
    weight_max: Optional[float] = None,
) -> np.ndarray:
    """Inverse-frequency weights from TRAIN labels only.

    weight[c] = N / (n_classes * count[c])

    Optional softening:
      - sqrt: square-root of inverse-frequency weights
      - sqrt_clip: sqrt then clip to [1/weight_max, weight_max] (or [0, weight_max] if max set)
      - clip: clip raw inverse-frequency to weight_max
      - none: raw inverse-frequency
    """
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    total = float(len(y))
    weights = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        if counts[c] > 0:
            weights[c] = total / (n_classes * counts[c])
        else:
            weights[c] = 1.0

    transform = (transform or "none").strip().lower()
    if transform in {"sqrt", "sqrt_clip"}:
        weights = np.sqrt(weights)
    if transform in {"clip", "sqrt_clip"} and weight_max is not None and weight_max > 0:
        weights = np.clip(weights, 1.0 / float(weight_max), float(weight_max))
    return weights


def oversample_fraud_train(
    df: pd.DataFrame,
    *,
    factor: float = 3.0,
    seed: int = 42,
    label_column: str = LABEL_COLUMN,
    fraud_label: int = 1,
) -> pd.DataFrame:
    """Train-only minority oversampling. Validation/test must never be passed here.

    ``factor`` multiplies the fraud row count (e.g. 3.0 => ~3x fraud examples).
    """
    if factor is None or float(factor) <= 1.0:
        return df.reset_index(drop=True)

    factor = float(factor)
    fraud = df[df[label_column] == fraud_label]
    if fraud.empty:
        return df.reset_index(drop=True)

    n_target = int(round(len(fraud) * factor))
    extra = n_target - len(fraud)
    if extra <= 0:
        return df.reset_index(drop=True)

    rng = np.random.RandomState(int(seed))
    sampled = fraud.sample(n=extra, replace=True, random_state=rng)
    out = pd.concat([df, sampled], ignore_index=True)
    return out.sample(frac=1.0, random_state=rng).reset_index(drop=True)


class JobTextDataset(Dataset):
    """Tokenised job-ad dataset for end-to-end BERT fine-tuning."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 256,
        record_ids: Optional[Sequence[str]] = None,
        titles: Optional[Sequence[str]] = None,
        companies: Optional[Sequence[str]] = None,
    ) -> None:
        self.texts = list(texts)
        self.labels = [int(x) for x in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.record_ids = list(record_ids) if record_ids is not None else [str(i) for i in range(len(self.texts))]
        self.titles = list(titles) if titles is not None else [""] * len(self.texts)
        self.companies = list(companies) if companies is not None else [""] * len(self.texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.labels[idx],
            "record_id": self.record_ids[idx],
            "title": self.titles[idx],
            "company": self.companies[idx],
            "row_index": idx,
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"]
        return item


def dataframe_to_text_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> JobTextDataset:
    return JobTextDataset(
        texts=df["model_text"].tolist(),
        labels=df[LABEL_COLUMN].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        record_ids=df[ID_COLUMN].astype(str).tolist(),
        titles=df["title"].astype(str).tolist() if "title" in df.columns else None,
        companies=df["company"].astype(str).tolist() if "company" in df.columns else None,
    )
