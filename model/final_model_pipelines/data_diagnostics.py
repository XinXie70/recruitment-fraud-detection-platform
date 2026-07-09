"""Dataset diagnostics for EMSCAD-style fake job data.

This script is intentionally dependency-free so it can run before installing the
training stack. It reports label imbalance, missing fields, duplicate text, and
simple shortcut-risk signals that can distort model evaluation.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_model_pipelines.shared_config import RAW_DATA_CANDIDATES, TEXT_FIELDS

LABEL_COL = "fraudulent"
BALANCED_MARKER_COL = "in_balanced_dataset"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _format_pct(count: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{count / total * 100:.2f}%"


def _resolve_input_path(raw_path: str | None) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    for path in RAW_DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset not found. Pass --csv /path/to/DataSet.csv or place DataSet.csv "
        f"in one of: {', '.join(str(p) for p in RAW_DATA_CANDIDATES)}"
    )


def _quantile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[int((len(sorted_values) - 1) * pct)]


def analyze_dataset(csv_path: Path) -> None:
    label_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    field_nonempty_by_label: dict[str, Counter[str]] = {field: Counter() for field in TEXT_FIELDS}
    lengths_by_label: defaultdict[str, list[int]] = defaultdict(list)
    combined_text_counts: Counter[str] = Counter()
    balanced_marker_cross: defaultdict[str, Counter[str]] = defaultdict(Counter)
    shortcut_checks: dict[str, Counter[str]] = {
        "company_profile_missing": Counter(),
        "requirements_missing": Counter(),
        "benefits_missing": Counter(),
        "salary_present": Counter(),
        "no_company_logo": Counter(),
        "has_questions": Counter(),
        "telecommuting": Counter(),
    }

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as file:
        reader = csv.DictReader(file)
        rows = 0
        columns = reader.fieldnames or []

        for row in reader:
            rows += 1
            label = (row.get(LABEL_COL) or "").strip()
            label_counts[label] += 1

            for column, value in row.items():
                if value is None or value.strip() == "":
                    missing_counts[column] += 1

            combined = _normalize_text(" ".join((row.get(field) or "") for field in TEXT_FIELDS))
            if combined:
                combined_text_counts[combined] += 1
            lengths_by_label[label].append(len(combined.split()))

            for field in TEXT_FIELDS:
                if (row.get(field) or "").strip():
                    field_nonempty_by_label[field][label] += 1

            balanced_marker_cross[label][(row.get(BALANCED_MARKER_COL) or "").strip()] += 1

            checks = {
                "company_profile_missing": not (row.get("company_profile") or "").strip(),
                "requirements_missing": not (row.get("requirements") or "").strip(),
                "benefits_missing": not (row.get("benefits") or "").strip(),
                "salary_present": bool((row.get("salary_range") or "").strip()),
                "no_company_logo": (row.get("has_company_logo") or "").strip() == "f",
                "has_questions": (row.get("has_questions") or "").strip() == "t",
                "telecommuting": (row.get("telecommuting") or "").strip() == "t",
            }
            for name, is_match in checks.items():
                if is_match:
                    shortcut_checks[name]["total"] += 1
                    if label in {"t", "1", "true", "True"}:
                        shortcut_checks[name]["fraud"] += 1

    print(f"Dataset: {csv_path}")
    print(f"Shape: {rows} rows x {len(columns)} columns")
    print("\nLabel distribution")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} ({_format_pct(count, rows)})")

    print("\nMissing fields")
    for column, count in missing_counts.most_common():
        print(f"  {column}: {count} ({_format_pct(count, rows)})")

    duplicate_rows = sum(count - 1 for count in combined_text_counts.values() if count > 1)
    print(f"\nDuplicate combined-text rows: {duplicate_rows}")

    print("\nText length by label")
    for label, values in sorted(lengths_by_label.items()):
        values.sort()
        mean = sum(values) / len(values) if values else 0
        print(
            f"  {label}: count={len(values)} mean={mean:.1f} "
            f"p10={_quantile(values, 0.10)} p50={_quantile(values, 0.50)} "
            f"p90={_quantile(values, 0.90)} max={values[-1] if values else 0}"
        )

    print("\nNon-empty text fields by label")
    for field, counts in field_nonempty_by_label.items():
        parts = []
        for label, total in label_counts.items():
            parts.append(f"{label}: {counts[label]}/{total} ({_format_pct(counts[label], total)})")
        print(f"  {field}: " + "; ".join(parts))

    print("\nShortcut-risk signals")
    for name, counts in shortcut_checks.items():
        total = counts["total"]
        fraud = counts["fraud"]
        print(f"  {name}: n={total}, fraud={fraud}, fraud_rate={_format_pct(fraud, total)}")

    if BALANCED_MARKER_COL in columns:
        print("\nBalanced marker cross-tab")
        print("  Warning: in_balanced_dataset is a split marker, not a model feature.")
        for label, counts in sorted(balanced_marker_cross.items()):
            print(f"  fraudulent={label}: {dict(counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data diagnostics for fake job datasets.")
    parser.add_argument("--csv", help="Path to DataSet.csv. Defaults to configured raw data locations.")
    args = parser.parse_args()
    analyze_dataset(_resolve_input_path(args.csv))


if __name__ == "__main__":
    main()
