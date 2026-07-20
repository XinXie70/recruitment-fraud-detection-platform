"""Group exact and near-duplicate EMSCAD advertisements.

Input:
    data/processed/emscad_processed_v1.csv

Outputs:
    data/processed/emscad_grouped_v1.csv
    data/diagnostics/duplicate_report_v1.md
"""

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import re


PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_DIR / "data/processed/emscad_processed_v1.csv"
OUTPUT_FILE = PROJECT_DIR / "data/processed/emscad_grouped_v1.csv"
REPORT_FILE = PROJECT_DIR / "data/diagnostics/duplicate_report_v1.md"

NEAR_DUPLICATE_THRESHOLD = 0.90


def normalise_for_exact_match(text):
    """Ignore case and whitespace differences when checking exact duplicates."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def get_title(text):
    """The title is the first non-empty line of the combined text."""
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line.casefold()).strip()
        if line:
            return line
    return ""


def get_word_set(text):
    """Return the unique lowercase words used for Jaccard similarity."""
    return set(re.findall(r"\b\w+\b", text.casefold()))


def jaccard_similarity(first, second):
    shared_words = len(first & second)
    all_words = len(first | second)
    return shared_words / all_words if all_words else 0.0


class Groups:
    """Join related row numbers into groups."""

    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def load_processed_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_FILE}\n"
            "Run prepare_data.py first."
        )

    with INPUT_FILE.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    required_columns = {"record_id", "combined_text", "label"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError("Processed dataset does not have the expected three columns")

    return rows


def group_duplicates():
    rows = load_processed_data()
    groups = Groups(len(rows))

    # Step 1: group exact duplicates.
    rows_by_exact_text = defaultdict(list)
    for row_number, row in enumerate(rows):
        exact_text = normalise_for_exact_match(row["combined_text"])
        rows_by_exact_text[exact_text].append(row_number)

    exact_duplicate_groups = [
        row_numbers
        for row_numbers in rows_by_exact_text.values()
        if len(row_numbers) > 1
    ]

    exact_conflicting_groups = [
        row_numbers
        for row_numbers in exact_duplicate_groups
        if len({rows[row_number]["label"] for row_number in row_numbers}) > 1
    ]
    if exact_conflicting_groups:
        raise ValueError(
            "Exact duplicates with different labels were found. "
            "Review them before removing duplicate copies."
        )

    for row_numbers in exact_duplicate_groups:
        first = row_numbers[0]
        for other in row_numbers[1:]:
            groups.join(first, other)

    # Keep the first row from each exact text. The processed file follows the
    # original EMSCAD row order, so this selection is reproducible.
    representatives = [row_numbers[0] for row_numbers in rows_by_exact_text.values()]

    # Step 2: compare the retained representatives for near duplicates.
    representatives_by_title = defaultdict(list)
    word_sets = {}

    for row_number in representatives:
        text = rows[row_number]["combined_text"]
        representatives_by_title[get_title(text)].append(row_number)
        word_sets[row_number] = get_word_set(text)

    checked_near_pairs = 0
    near_duplicate_pairs = []

    for same_title_rows in representatives_by_title.values():
        for first, second in combinations(same_title_rows, 2):
            checked_near_pairs += 1
            similarity = jaccard_similarity(word_sets[first], word_sets[second])
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                groups.join(first, second)
                near_duplicate_pairs.append((first, second, similarity))

    # Step 3: collect final groups using only the retained representatives.
    members_by_root = defaultdict(list)
    for row_number in representatives:
        members_by_root[groups.find(row_number)].append(row_number)

    final_groups = sorted(members_by_root.values(), key=lambda members: members[0])
    group_id_by_row = {}
    group_members_by_id = {}

    for group_number, members in enumerate(final_groups, start=1):
        group_id = f"group_{group_number:05d}"
        group_members_by_id[group_id] = members
        for row_number in members:
            group_id_by_row[row_number] = group_id

    # Step 4: identify groups containing both labels.
    conflicting_groups = []
    for group_id, members in group_members_by_id.items():
        labels = {rows[row_number]["label"] for row_number in members}
        if len(labels) > 1:
            conflicting_groups.append((group_id, members))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["record_id", "combined_text", "label", "group_id"],
        )
        writer.writeheader()
        for row_number in representatives:
            row = rows[row_number]
            writer.writerow(
                {
                    "record_id": row["record_id"],
                    "combined_text": row["combined_text"],
                    "label": row["label"],
                    "group_id": group_id_by_row[row_number],
                }
            )

    duplicate_final_groups = [members for members in final_groups if len(members) > 1]
    largest_group = max((len(members) for members in final_groups), default=0)
    removed_exact_copies = len(rows) - len(representatives)

    report_lines = [
        "# Duplicate Grouping Report v1",
        "",
        f"- Input rows: **{len(rows):,}**",
        f"- Exact duplicate copies removed: **{removed_exact_copies:,}**",
        f"- Rows kept after exact deduplication: **{len(representatives):,}**",
        f"- Final groups: **{len(final_groups):,}**",
        f"- Groups containing more than one row: **{len(duplicate_final_groups):,}**",
        f"- Exact duplicate groups: **{len(exact_duplicate_groups):,}**",
        f"- Rows inside exact duplicate groups: **{sum(len(group) for group in exact_duplicate_groups):,}**",
        f"- Near-duplicate pairs found: **{len(near_duplicate_pairs):,}** from {checked_near_pairs:,} checked pairs",
        f"- Largest final group: **{largest_group} rows**",
        f"- Groups with both labels: **{len(conflicting_groups)}**",
        "",
        "## Rules used",
        "",
        "- Exact duplicate: same combined text after lowercasing and collapsing whitespace.",
        f"- Near duplicate: same cleaned title and word-set Jaccard similarity >= {NEAR_DUPLICATE_THRESHOLD:.2f}.",
        "- The first record in source order is kept for each exact text.",
        "- Different near-duplicate texts are retained and assigned the same group ID.",
        "- Labels of retained records are not changed.",
        "",
        "## Label conflicts",
        "",
    ]

    if conflicting_groups:
        report_lines.extend([
            "These groups contain both label 0 and label 1 and should be reviewed before splitting.",
            "",
            "| Group | Size | Record IDs | Labels |",
            "|---|---:|---|---|",
        ])
        for group_id, members in conflicting_groups:
            record_ids = ", ".join(rows[number]["record_id"] for number in members)
            labels = ", ".join(rows[number]["label"] for number in members)
            report_lines.append(
                f"| {group_id} | {len(members)} | {record_ids} | {labels} |"
            )
    else:
        report_lines.append("No groups contain conflicting labels.")

    report_lines.extend([
        "",
        "## Limitation",
        "",
        "Near-duplicate checking only compares advertisements with the same cleaned title. Similar templates with different titles may be missed.",
        "",
    ])

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Input rows: {len(rows):,}")
    print(f"Exact duplicate copies removed: {removed_exact_copies:,}")
    print(f"Rows kept: {len(representatives):,}")
    print(f"Final groups: {len(final_groups):,}")
    print(f"Groups with multiple rows: {len(duplicate_final_groups):,}")
    print(f"Groups with both labels: {len(conflicting_groups)}")
    print(f"Grouped data saved to: {OUTPUT_FILE}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    group_duplicates()
