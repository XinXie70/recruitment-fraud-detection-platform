"""Create the fixed group-aware Train, Validation, and Test files."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

from analyse_split_options import allocate_groups, load_groups


PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_DIR / "data/processed/emscad_grouped_v1.csv"
OUTPUT_DIR = PROJECT_DIR / "data/splits"
ASSIGNMENT_FILE = OUTPUT_DIR / "split_assignments_v1.csv"
REPORT_FILE = PROJECT_DIR / "data/diagnostics/split_report_v1.md"

RATIOS = [0.70, 0.15, 0.15]
SPLITS = ["train", "validation", "test"]
SEED = 42


def main():
    group_data, expected_rows = load_groups()
    generated_assignment = allocate_groups(group_data, RATIOS, SEED)

    saved_assignments = {}
    if ASSIGNMENT_FILE.exists():
        with ASSIGNMENT_FILE.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                saved_assignments[row["record_id"]] = (
                    row["group_id"], row["split"]
                )

    rows_by_split = defaultdict(list)
    with INPUT_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        for row in reader:
            if saved_assignments:
                if row["record_id"] not in saved_assignments:
                    raise ValueError(
                        f"Record missing from saved assignments: {row['record_id']}"
                    )
                saved_group, split = saved_assignments[row["record_id"]]
                if saved_group != row["group_id"]:
                    raise ValueError(
                        f"Group mismatch for record: {row['record_id']}"
                    )
            else:
                split = generated_assignment[row["group_id"]]
            rows_by_split[split].append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not saved_assignments:
        with ASSIGNMENT_FILE.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["record_id", "group_id", "split"])
            for split in SPLITS:
                for row in rows_by_split[split]:
                    writer.writerow([row["record_id"], row["group_id"], split])

    if saved_assignments and len(saved_assignments) != expected_rows:
        raise ValueError("Saved assignment count does not match the dataset.")

    for split in SPLITS:
        output_file = OUTPUT_DIR / f"{split}.csv"
        with output_file.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_by_split[split])

    all_rows = [row for split in SPLITS for row in rows_by_split[split]]
    record_ids = [row["record_id"] for row in all_rows]
    if len(all_rows) != expected_rows or len(set(record_ids)) != expected_rows:
        raise ValueError("Records are missing or assigned more than once.")

    group_sets = {
        split: {row["group_id"] for row in rows_by_split[split]}
        for split in SPLITS
    }
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1:]:
            if group_sets[first] & group_sets[second]:
                raise ValueError(f"A group crosses {first} and {second}.")

    report = [
        "# Final Split Report v1",
        "",
        "- Selected ratio: **70% Train / 15% Validation / 15% Test**",
        "- Random seed: **42**",
        "- Allocation unit: complete duplicate group",
        "",
        "| Split | Rows | Legitimate | Fraudulent | Groups | Fraud groups |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        rows = rows_by_split[split]
        labels = Counter(int(row["label"]) for row in rows)
        fraud_groups = {
            row["group_id"] for row in rows if int(row["label"]) == 1
        }
        report.append(
            f"| {split.title()} | {len(rows):,} | {labels[0]:,} | "
            f"{labels[1]:,} | {len(group_sets[split]):,} | {len(fraud_groups):,} |"
        )

    report.extend([
        "",
        f"All **{expected_rows:,}** records were assigned exactly once.",
        "No duplicate group crosses between splits.",
        "",
    ])
    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print(f"Created fixed splits in: {OUTPUT_DIR}")
    print(f"Fixed assignment file: {ASSIGNMENT_FILE}")
    print(f"Validation report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
