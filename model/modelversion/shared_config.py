"""Shared paths and constants for final_model_pipelines."""

from pathlib import Path

PIPELINES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINES_ROOT.parent

# Data directory (self-contained, no dependency on data_cleaning/)
DATA_DIR = PIPELINES_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.csv"

# Raw data: prefer pipelines/data/, then repository root
RAW_DATA_CANDIDATES = (
    DATA_DIR / "DataSet.csv",
    PROJECT_ROOT / "DataSet.csv",
)

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
TEXT_COL = "combined_text"
LABEL_COL = "fraudulent"
POSITIVE_LABEL = 1

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1


def resolve_raw_data_path() -> Path:
    for path in RAW_DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Raw dataset not found. Place DataSet.csv in one of the following locations:\n"
        f"  - {RAW_DATA_CANDIDATES[0]}\n"
        f"  - {RAW_DATA_CANDIDATES[1]}"
    )
