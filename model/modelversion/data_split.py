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
from final_model_pipelines.text_utils import (
    deduplicate_by_combined_text,
    preprocess_raw_dataframe,
)


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


def _split_paths() -> tuple[Path, Path, Path]:
    return SPLIT_DIR / "train.csv", SPLIT_DIR / "val.csv", SPLIT_DIR / "test.csv"


def clear_prepared_artifacts(*, clear_cleaned: bool = True) -> None:
    """Remove cached splits (and optionally cleaned_data.csv) so prepare_data can rebuild."""
    for path in _split_paths():
        if not path.exists():
            continue
        try:
            path.unlink()
            print(f"Removed: {path}")
        except OSError as exc:
            print(f"Could not remove {path} ({exc}); will overwrite on save.")
    if clear_cleaned and CLEANED_DATA_PATH.exists():
        try:
            CLEANED_DATA_PATH.unlink()
            print(f"Removed: {CLEANED_DATA_PATH}")
        except OSError as exc:
            print(f"Could not remove {CLEANED_DATA_PATH} ({exc}); will overwrite on save.")


def build_cleaned_dataframe(raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Clean raw rows, build combined_text, then deduplicate by normalized text.

    Preference on duplicate keys: keep the row with more non-empty text fields.
    """
    if raw_df is None:
        raw_path = resolve_raw_data_path()
        print(f"Cleaning raw data: {raw_path}")
        raw_df = pd.read_csv(raw_path)
    df = preprocess_raw_dataframe(raw_df)
    return deduplicate_by_combined_text(df)


def load_or_create_splits(
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train / val / test; if missing (or force=True), clean, dedupe, and split.

    Data flow:
      DataSet.csv → clean → combined_text → dedupe → cleaned_data.csv
        → stratified split → train/val/test.csv
    """
    train_path, val_path, test_path = _split_paths()

    if force:
        clear_prepared_artifacts(clear_cleaned=True)

    if (
        not force
        and train_path.exists()
        and val_path.exists()
        and test_path.exists()
    ):
        return pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Always rebuild from raw when forcing, or when cleaned_data is missing.
    if force or not CLEANED_DATA_PATH.exists():
        df = build_cleaned_dataframe()
    else:
        print(f"Loading cleaned data: {CLEANED_DATA_PATH}")
        df = deduplicate_by_combined_text(pd.read_csv(CLEANED_DATA_PATH))

    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Saved cleaned data: {CLEANED_DATA_PATH} ({len(df)} rows)")

    return split_and_save(df, output_dir=SPLIT_DIR)
