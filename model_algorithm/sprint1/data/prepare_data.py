#Prepare EMSCAD data for Sprint 1.

from __future__ import annotations
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict
import pandas as pd
from sklearn.model_selection import train_test_split
BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "emscad_v1.csv"
SPLIT_DIR = BASE_DIR / "splits"
RANDOM_SEED = 42
TEXT_FIELDS = (
    "title","company_profile","description","requirements","benefits",)

def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()
def parse_fraud_label(value: object) -> int:
    label = str(value).strip().lower()
    label_mapping = {"0": 0,"f": 0,"false": 0,"1": 1,"t": 1,"true": 1,}
    if label not in label_mapping:
        raise ValueError(f"Unsupported fraudulent label: {value!r}")
    return label_mapping[label]


def merge_text_columns(row: pd.Series) -> str:
    cleaned_parts = []
    for column in TEXT_FIELDS:
        cleaned = normalize_text(row.get(column, ""))
        if cleaned:
            cleaned_parts.append(cleaned)
    return "\n".join(cleaned_parts)


def load_and_prepare_dataset(csv_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)
    required_columns = [*TEXT_FIELDS, "fraudulent"]
    missing_columns = [
        column
        for column in required_columns
        if column not in raw_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Required columns are missing from the dataset: {missing_columns}"
        )

    prepared_df = pd.DataFrame(
        {
            "record_id": [
                f"emscad_{index + 1:05d}"
                for index in range(len(raw_df))
            ],
            "combined_text": raw_df.apply(
                merge_text_columns,axis=1,
            ),
            "label": raw_df["fraudulent"].map(parse_fraud_label),
        }
    )
    return prepared_df


def remove_duplicate_text(
    dataframe: pd.DataFrame,) -> tuple[pd.DataFrame, int]:
    original_size = len(dataframe)
    deduplicated = (
        dataframe
        .drop_duplicates(
            subset="combined_text",keep="first",
        )
        .reset_index(drop=True)
    )
    removed_count = original_size - len(deduplicated)
    return deduplicated, removed_count


def create_dataset_splits(
    dataframe: pd.DataFrame,
    random_seed: int = RANDOM_SEED,) -> Dict[str, pd.DataFrame]:
    train_df, temporary_df = train_test_split(
        dataframe,test_size=0.30,stratify=dataframe["label"],random_state=random_seed,
    )

    validation_df, test_df = train_test_split(
        temporary_df,test_size=0.50,stratify=temporary_df["label"],random_state=random_seed,
    )

    return {
        "train": train_df.reset_index(drop=True),
        "validation": validation_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def describe_split(dataframe: pd.DataFrame) -> dict:
    fraud_mask = dataframe["label"].eq(1)

    return {
        "n": len(dataframe),"fraud": int(fraud_mask.sum()),"fraud_rate": float(fraud_mask.mean()),
    }


def save_splits(
    splits: Dict[str, pd.DataFrame],output_dir: Path,) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary = {}
    for split_name, split_df in splits.items():
        output_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(
            output_path,
            index=False,
        )
        stats = describe_split(split_df)
        split_summary[split_name] = stats
        print(
            f"Saved {output_path.name}: "f"n={stats['n']}, "
            f"fraud={stats['fraud']}, "
            f"fraud_rate={stats['fraud_rate']:.4f}"
        )
    return split_summary


def save_metadata(
    output_dir: Path,duplicate_count: int,empty_count: int,split_summary: dict,total_rows: int,
) -> None:
    metadata = {
        "seed": RANDOM_SEED,
        "protocol": "70/15/15 stratified",
        "source": SOURCE_FILE.name,
        "exact_duplicates_removed": duplicate_count,
        "empty_combined_text_rows_kept": empty_count,
        "split_counts": split_summary,
        "n_total_after_dedup": total_rows,
        "columns": ["record_id","combined_text","label",],
        "cleaning": [
            "missing text -> empty string",
            "HTML entity decode","HTML tag removal","Unicode NFKC",
            "newline / whitespace normalisation",
            "exact duplicate combined_text dropped (keep first)",],
    }
    metadata_path = output_dir / "split_meta.json"
    metadata_path.write_text(
        json.dumps(
            metadata,indent=2,ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SOURCE_FILE}"
        )
    print(f"Loading dataset: {SOURCE_FILE}")
    dataframe = load_and_prepare_dataset(SOURCE_FILE)
    fraud_count = int(dataframe["label"].eq(1).sum())
    print(
        f"Prepared rows: {len(dataframe)} | "
        f"Fraud cases: {fraud_count}"
    )
    empty_count = int(
        dataframe["combined_text"]
        .astype(str).str.strip().eq("").sum()
    )
    print(
        f"Empty combined_text rows: {empty_count}"
    )
    dataframe, duplicate_count = remove_duplicate_text(dataframe)
    print(
        f"Removed exact duplicates: {duplicate_count}"
    )
    print(
        f"Rows after deduplication: {len(dataframe)} | "
        f"Fraud cases: {int(dataframe['label'].eq(1).sum())}"
    )
    dataset_splits = create_dataset_splits(dataframe,RANDOM_SEED,)
    split_summary = save_splits(dataset_splits,SPLIT_DIR,)
    save_metadata(
        output_dir=SPLIT_DIR,
        duplicate_count=duplicate_count,
        empty_count=empty_count,
        split_summary=split_summary,
        total_rows=len(dataframe),
    )


if __name__ == "__main__":
    main()