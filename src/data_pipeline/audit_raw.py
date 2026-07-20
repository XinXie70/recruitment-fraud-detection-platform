"""Simple read-only audit for the EMSCAD CSV.

This script uses only the Python standard library. It reads the raw file and
writes one Markdown report. It does not change the source data.
"""

import csv
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
from pathlib import Path
import re
import statistics

from prepare_data import clean_text, combine_text_fields, convert_label


PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_DIR / "data/raw/emscad_v1.csv"
OUTPUT_FILE = PROJECT_DIR / "data/diagnostics/raw_audit_v1.md"

def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def percentile(values, percentage):
    """Return a simple linear percentile without NumPy."""
    values = sorted(values)
    position = (len(values) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


class Groups:
    """Small helper for joining rows into near-duplicate groups."""

    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, first, second):
        first = self.find(first)
        second = self.find(second)
        if first != second:
            self.parent[second] = first


def find_near_duplicates(titles, texts):
    """Find a preliminary set of similar ads using an easy-to-explain rule.

    Only ads with the same cleaned title are compared. Two ads are considered
    near-duplicates when the Jaccard similarity of their word sets is at least
    0.90. Very large title groups are skipped to keep this student audit quick.
    """
    rows_by_title = defaultdict(list)
    word_sets = []
    for row_number, (title, text) in enumerate(zip(titles, texts)):
        rows_by_title[title.lower()].append(row_number)
        word_sets.append(set(re.findall(r"\b\w+\b", text.lower())))

    groups = Groups(len(texts))
    checked_pairs = 0
    similar_pairs = 0
    skipped_title_groups = 0

    for rows in rows_by_title.values():
        if len(rows) < 2:
            continue
        if len(rows) > 100:
            skipped_title_groups += 1
            continue
        for first, second in combinations(rows, 2):
            if texts[first].lower() == texts[second].lower():
                continue
            checked_pairs += 1
            shared = len(word_sets[first] & word_sets[second])
            total = len(word_sets[first] | word_sets[second])
            similarity = shared / total if total else 0
            if similarity >= 0.90:
                groups.join(first, second)
                similar_pairs += 1

    members = defaultdict(list)
    for row_number in range(len(texts)):
        members[groups.find(row_number)].append(row_number)
    near_groups = [rows for rows in members.values() if len(rows) > 1]
    return near_groups, checked_pairs, similar_pairs, skipped_title_groups


def run_audit():
    missing = Counter()
    labels = Counter()
    texts = []
    titles = []

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        for row in reader:
            for column in columns:
                if not row[column].strip():
                    missing[column] += 1
            labels[convert_label(row["fraudulent"])] += 1
            texts.append(combine_text_fields(row))
            titles.append(clean_text(row["title"]))

    character_lengths = [len(text) for text in texts]
    word_lengths = [len(text.split()) for text in texts]

    exact_counts = Counter(
        re.sub(r"\s+", " ", text.casefold()).strip()
        for text in texts
        if text
    )
    exact_groups = [count for count in exact_counts.values() if count > 1]

    near_groups, checked_pairs, similar_pairs, skipped = find_near_duplicates(titles, texts)

    lines = [
        "# Raw Data Audit v1",
        "",
        f"- File: `data/raw/emscad_v1.csv`",
        f"- SHA-256: `{file_sha256(INPUT_FILE)}`",
        f"- Rows: **{len(texts):,}**",
        f"- Columns: **{len(columns)}**",
        "",
        "## Label distribution",
        "",
        f"- Legitimate (0): {labels[0]:,} ({labels[0] / len(texts):.2%})",
        f"- Fraudulent (1): {labels[1]:,} ({labels[1] / len(texts):.2%})",
        "",
        "## Missing values",
        "",
        "| Column | Missing | Percentage |",
        "|---|---:|---:|",
    ]

    for column in columns:
        lines.append(
            f"| {column} | {missing[column]:,} | {missing[column] / len(texts):.2%} |"
        )

    lines.extend([
        "",
        "## Combined text length",
        "",
        "| Measure | Characters | Words |",
        "|---|---:|---:|",
        f"| Mean | {statistics.mean(character_lengths):,.1f} | {statistics.mean(word_lengths):,.1f} |",
        f"| Median | {statistics.median(character_lengths):,.1f} | {statistics.median(word_lengths):,.1f} |",
        f"| 95th percentile | {percentile(character_lengths, 0.95):,.1f} | {percentile(word_lengths, 0.95):,.1f} |",
        f"| Maximum | {max(character_lengths):,} | {max(word_lengths):,} |",
        "",
        "## Duplicate check",
        "",
        f"- Exact duplicate groups: **{len(exact_groups):,}**",
        f"- Rows inside exact duplicate groups: **{sum(exact_groups):,}**",
        f"- Preliminary near-duplicate groups: **{len(near_groups):,}**",
        f"- Near-duplicate pairs found: **{similar_pairs:,}** from {checked_pairs:,} checked pairs",
        f"- Large same-title groups skipped: **{skipped}**",
        "",
        "Near-duplicate rule: ads must have the same cleaned title and at least 0.90 Jaccard similarity between their word sets. This is a preliminary audit, not the final split rule.",
        "",
    ])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit complete: {len(texts):,} rows")
    print(f"Report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_audit()
