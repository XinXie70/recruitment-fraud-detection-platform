"""
Data preparation entry point: cleaning + deduplication + stratified splitting.

Self-contained within final_model_pipelines; no dependency on data_cleaning/.

Usage (from repository root / model parent of final_model_pipelines):
    python final_model_pipelines/prepare_data.py
    python final_model_pipelines/prepare_data.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.data_split import load_or_create_splits
from final_model_pipelines.shared_config import CLEANED_DATA_PATH, SPLIT_DIR, resolve_raw_data_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean DataSet.csv, dedupe by combined_text, then stratified split."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete cleaned_data.csv and splits/, then rebuild from DataSet.csv.",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("final_model_pipelines — Data Cleaning, Dedup, and Splitting")
    print("=" * 50)
    print(f"Raw data: {resolve_raw_data_path()}")
    load_or_create_splits(force=args.force)
    print(f"\nCleaned data: {CLEANED_DATA_PATH}")
    print(f"Split directory: {SPLIT_DIR}")


if __name__ == "__main__":
    main()
