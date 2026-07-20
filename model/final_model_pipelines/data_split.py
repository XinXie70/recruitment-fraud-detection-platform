"""Stratified train / val / test splitting (self-contained, no dependency on data_cleaning)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from final_model_pipelines.shared_config import (
    CLEANED_DATA_PATH,
    DATA_DIR,
    LABEL_COL,
    RANDOM_STATE,
    SPLIT_DIR,
    TEST_SIZE,
    VAL_SIZE,
    resolve_raw_data_path,
)
from final_model_pipelines.text_utils import preprocess_raw_dataframe


def _fraud_ratio(df: pd.DataFrame, label_col: str = LABEL_COL) -> float:
    return float(df[label_col].mean())


def _print_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = LABEL_COL,
) -> None:
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"Train set: {len(train_df)} rows (fraud ratio {_fraud_ratio(train_df, label_col):.4f})")
    print(f"Validation set: {len(val_df)} rows (fraud ratio {_fraud_ratio(val_df, label_col):.4f})")
    print(f"Test set: {len(test_df)} rows (fraud ratio {_fraud_ratio(test_df, label_col):.4f})")
    print(
        f"Split ratios: train={len(train_df) / total:.1%}, "
        f"val={len(val_df) / total:.1%}, test={len(test_df) / total:.1%}"
    )


def stratified_train_val_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
    label_col: str = LABEL_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.0")

    y = df[label_col]
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val_df[label_col],
    )
    return train_df, val_df, test_df


def split_and_save(
    cleaned_df: pd.DataFrame,
    output_dir: Path | None = None,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = output_dir or SPLIT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = stratified_train_val_test_split(
        cleaned_df, test_size=test_size, val_size=val_size, random_state=random_state
    )
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("\nStratified split complete:")
    _print_split_stats(train_df, val_df, test_df)
    print(f"Saved: {output_dir / 'train.csv'}")
    print(f"Saved: {output_dir / 'val.csv'}")
    print(f"Saved: {output_dir / 'test.csv'}")
    return train_df, val_df, test_df


def load_or_create_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train / val / test; if missing, clean from raw CSV and split automatically.

    Data flow:
      DataSet.csv → clean → cleaned_data.csv → stratified split → train/val/test.csv
    """
    train_path = SPLIT_DIR / "train.csv"
    val_path = SPLIT_DIR / "val.csv"
    test_path = SPLIT_DIR / "test.csv"

    if train_path.exists() and val_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CLEANED_DATA_PATH.exists():
        df = pd.read_csv(CLEANED_DATA_PATH)
    else:
        raw_path = resolve_raw_data_path()
        print(f"Cleaning raw data: {raw_path}")
        df = preprocess_raw_dataframe(pd.read_csv(raw_path))
        df.to_csv(CLEANED_DATA_PATH, index=False)
        print(f"Saved cleaned data: {CLEANED_DATA_PATH}")

    return split_and_save(df, output_dir=SPLIT_DIR)
