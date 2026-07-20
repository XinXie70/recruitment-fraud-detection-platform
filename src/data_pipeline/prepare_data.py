"""Prepare the shared EMSCAD text dataset used by all models.

Input:
    data/raw/emscad_v1.csv

Output:
    data/processed/emscad_processed_v1.csv
"""

import csv
import html
from pathlib import Path
import re
import unicodedata


PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_DIR / "data/raw/emscad_v1.csv"
OUTPUT_FILE = PROJECT_DIR / "data/processed/emscad_processed_v1.csv"

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]


def clean_text(value):
    """Apply the shared basic cleaning rules to one text field."""
    if value is None:
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def combine_text_fields(row):
    """Combine the five agreed EMSCAD text fields in a fixed order."""
    cleaned_fields = [clean_text(row.get(column, "")) for column in TEXT_COLUMNS]
    return "\n".join(text for text in cleaned_fields if text)


def convert_label(value):
    """Convert EMSCAD's f/t label to the shared 0/1 label."""
    value = str(value).strip().lower()
    if value == "f":
        return 0
    if value == "t":
        return 1
    raise ValueError(f"Unexpected fraudulent label: {value!r}")


def prepare_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    label_counts = {0: 0, 1: 0}
    empty_text_count = 0

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)

        required_columns = set(TEXT_COLUMNS + ["fraudulent"])
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=["record_id", "combined_text", "label"],
            )
            writer.writeheader()

            for row_number, row in enumerate(reader, start=1):
                combined_text = combine_text_fields(row)
                label = convert_label(row["fraudulent"])
                record_id = f"emscad_{row_number:05d}"

                if not combined_text:
                    empty_text_count += 1

                writer.writerow(
                    {
                        "record_id": record_id,
                        "combined_text": combined_text,
                        "label": label,
                    }
                )

                row_count += 1
                label_counts[label] += 1

    print(f"Processed rows: {row_count:,}")
    print(f"Legitimate (0): {label_counts[0]:,}")
    print(f"Fraudulent (1): {label_counts[1]:,}")
    print(f"Empty combined text: {empty_text_count}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_data()
