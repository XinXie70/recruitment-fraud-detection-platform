"""
Data preparation entry point: cleaning + stratified splitting.

Self-contained within final_model_pipelines; no dependency on data_cleaning/.

Usage (from repository root):
    python final_model_pipelines/prepare_data.py
"""

from __future__ import annotations

from final_model_pipelines.data_split import load_or_create_splits
from final_model_pipelines.shared_config import CLEANED_DATA_PATH, SPLIT_DIR, resolve_raw_data_path


def main() -> None:
    print("=" * 50)
    print("final_model_pipelines — Data Cleaning and Splitting")
    print("=" * 50)
    print(f"Raw data: {resolve_raw_data_path()}")
    load_or_create_splits()
    print(f"\nCleaned data: {CLEANED_DATA_PATH}")
    print(f"Split directory: {SPLIT_DIR}")


if __name__ == "__main__":
    main()
