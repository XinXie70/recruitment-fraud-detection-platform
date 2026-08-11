#Split paper-aligned data for EMSCAD data
from __future__ import annotations
import argparse, html, json, re, unicodedata, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import train_test_split
RAW_PATH = Path(__file__).resolve().parent
DATA_PATH = RAW_PATH.parent
RANDOM_SEED = 42
ID_COLUMN = "record_id"
LABEL_COLUMN = "label"
TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
SECTION_WORD_CAPS = {
    "title": 20,
    "company_profile": 80,
    "description": 300,
    "requirements": 120,
    "benefits": 80,
}

def clean_text(value) -> str:
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
    label = str(value).strip().lower()
    mapping = {"0": 0, "f": 0, "false": 0, "1": 1, "t": 1, "true": 1}
    if label not in mapping:
        raise ValueError(f"Unexpected fraudulent label: {value!r}")
    return mapping[label]
def combine_text_fields(row: pd.Series) -> str:
    parts = [clean_text(row.get(column, "")) for column in TEXT_COLUMNS]
    return "\n".join(part for part in parts if part)
def cap_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text)
    return " ".join(words[:max_words])

def build_model_text(combined_text: str) -> str:
    if combined_text is None or (isinstance(combined_text, float) and pd.isna(combined_text)):
        return ""
    lines = [line.strip() for line in str(combined_text).split("\n") if line.strip()]
    if not lines:
        return ""
    parts: List[str] = [f"[TITLE] {cap_words(lines[0], SECTION_WORD_CAPS['title'])}"]
    if len(lines) == 1:
        return "\n".join(parts)
    remaining = lines[1:]
    if len(remaining) == 1:
        parts.append(f"[DESCRIPTION] {cap_words(remaining[0], SECTION_WORD_CAPS['description'])}")
        return "\n".join(parts)
    parts.append(f"[COMPANY PROFILE] {cap_words(remaining[0], SECTION_WORD_CAPS['company_profile'])}")
    body = remaining[1:]
    if not body:
        return "\n".join(parts)
    if len(body) == 1:
        parts.append(f"[DESCRIPTION] {cap_words(body[0], SECTION_WORD_CAPS['description'])}")
        return "\n".join(parts)

    first_cut = max(1, len(body) // 3)
    second_cut = max(first_cut + 1, (2 * len(body)) // 3)
    description = " ".join(body[:first_cut])
    requirements = " ".join(body[first_cut:second_cut])
    benefits = " ".join(body[second_cut:])
    parts.append(f"[DESCRIPTION] {cap_words(description, SECTION_WORD_CAPS['description'])}")

    if requirements.strip():
        parts.append(f"[REQUIREMENTS] {cap_words(requirements, SECTION_WORD_CAPS['requirements'])}")
    if benefits.strip():
        parts.append(f"[BENEFITS] {cap_words(benefits, SECTION_WORD_CAPS['benefits'])}")
    return "\n".join(parts)

def read_raw_data(file_path: Path) -> pd.DataFrame:
    raw_data = pd.read_csv(file_path)
    required_columns = TEXT_COLUMNS + ["fraudulent"]
    missing_columns = [column for column in required_columns if column not in raw_data.columns]
    if missing_columns:
        raise ValueError(f"Raw CSV missing columns: {missing_columns}")
    rows = []

    for index, row in raw_data.iterrows():
        record_id = f"emscad_{index + 1:05d}"
        combined_text = combine_text_fields(row)
        model_text = build_model_text(combined_text)
        if not model_text.strip():
            model_text = combined_text
        rows.append({
            ID_COLUMN: record_id,
            "combined_text": combined_text,
            "model_text": model_text,
            LABEL_COLUMN: convert_label(row["fraudulent"]),
            "title": str(model_text).split("\n", 1)[0][:120],
            "company": "",
        })

    return pd.DataFrame(rows)


def create_assignment(dataframe: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    labels = dataframe[LABEL_COLUMN]
    indices = np.arange(len(dataframe))
    training_pool, test_indices = train_test_split(indices, test_size=0.20, stratify=labels, random_state=seed)
    training_indices, validation_indices = train_test_split(training_pool, test_size=0.10, stratify=labels.iloc[training_pool], random_state=seed)
    split_labels = pd.Series("", index=dataframe.index)
    split_labels.iloc[training_indices] = "train"
    split_labels.iloc[validation_indices] = "validation"
    split_labels.iloc[test_indices] = "test"
    return pd.DataFrame({ID_COLUMN: dataframe[ID_COLUMN].astype(str), "seed": seed, "split": split_labels.astype(str)})

def save_splits(dataframe: pd.DataFrame, assignment: pd.DataFrame, output_path: Path) -> Dict:
    output_path.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(output_path / "assignments.csv.gz", index=False, compression="gzip")
    merged_data = assignment.merge(dataframe, on=ID_COLUMN, how="left", validate="one_to_one")
    split_counts = {}

    for split_name in ("train", "validation", "test"):
        split_data = merged_data[merged_data["split"] == split_name].copy().reset_index(drop=True)
        split_data.to_csv(output_path / f"{split_name}.csv.gz", index=False, compression="gzip")
        split_counts[split_name] = {"n": int(len(split_data)), "fraud": int((split_data[LABEL_COLUMN] == 1).sum())}
    metadata = {
        "seed": int(assignment["seed"].iloc[0]),
        "protocol": "80/20 + 10% val-of-train",
        "split_counts": split_counts,
        "n_total": int(len(dataframe)),
    }

    (output_path / "split_meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata

def save_processed_data(dataframe: pd.DataFrame) -> Path:
    processed_path = DATA_PATH / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    output_file = processed_path / "emscad_condition_a_no_dedup_input_v1.csv.gz"
    dataframe[[ID_COLUMN, "combined_text", LABEL_COLUMN]].to_csv(output_file, index=False, compression="gzip")
    return output_file

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_PATH / "emscad_v1.csv")
    parser.add_argument("--out", type=Path, default=DATA_PATH / "splits")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--also_save_processed", action="store_true", help="Also write ../processed/emscad_condition_a_no_dedup_input_v1.csv.gz")
    return parser.parse_args()

def main() -> None:
    arguments = parse_arguments()
    if not arguments.raw.exists():
        raise FileNotFoundError(f"Raw EMSCAD not found: {arguments.raw}\nPlace emscad_v1.csv next to this script first.")
    print(f"Loading raw: {arguments.raw}")
    dataframe = read_raw_data(arguments.raw)
    print(f"Built frame n={len(dataframe)} fraud={int((dataframe[LABEL_COLUMN] == 1).sum())}")
    if arguments.also_save_processed:
        processed_file = save_processed_data(dataframe)
        print(f"Wrote processed: {processed_file}")
    assignment = create_assignment(dataframe, arguments.seed)
    metadata = save_splits(dataframe, assignment, arguments.out)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Splits written to: {arguments.out}")
if __name__ == "__main__":
    main()