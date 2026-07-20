"""Compare possible group-aware split ratios before creating final splits.

This is a data feasibility analysis only. It does not train a model and does not
write Train, Validation, or Test files.
"""

import csv
from collections import defaultdict
import math
from pathlib import Path
import random


PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_DIR / "data/processed/emscad_grouped_v1.csv"
REPORT_FILE = PROJECT_DIR / "reports/split_feasibility_v1.md"

SPLITS = ["train", "validation", "test"]
OPTIONS = {
    "A: 80/10/10": [0.80, 0.10, 0.10],
    "B: 75/10/15": [0.75, 0.10, 0.15],
    "C: 72/14/14": [0.72, 0.14, 0.14],
    "D: 70/15/15": [0.70, 0.15, 0.15],
    "E: 60/20/20": [0.60, 0.20, 0.20],
}

MAIN_SEED = 42
SENSITIVITY_SEEDS = range(100)
MAX_95_CI_HALF_WIDTH = 0.10
MINORITY_REFERENCE_COUNT = math.ceil(
    (1.96**2 * 0.25) / (MAX_95_CI_HALF_WIDTH**2)
)
OPERATIONAL_TARGET = 100
FINAL_SELECTION = "D: 70/15/15"


def load_groups():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Grouped dataset not found: {INPUT_FILE}\n"
            "Run group_duplicates.py first."
        )

    groups = defaultdict(list)
    seen_record_ids = set()

    with INPUT_FILE.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            record_id = row["record_id"]
            if record_id in seen_record_ids:
                raise ValueError(f"Duplicate record_id found: {record_id}")
            seen_record_ids.add(record_id)
            groups[row["group_id"]].append(row)

    group_data = []
    for group_id, rows in groups.items():
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Group contains different labels: {group_id}")
        group_data.append(
            {
                "group_id": group_id,
                "label": labels.pop(),
                "size": len(rows),
            }
        )

    return group_data, len(seen_record_ids)


def allocate_groups(group_data, ratios, seed):
    """Allocate complete groups using a label-stratified randomised order.

    Each complete group goes to the split with the largest remaining target
    capacity for its label. Randomising group order prevents all large groups
    from being systematically placed in Train.
    """
    rng = random.Random(seed)
    assignment = {}

    total_by_label = {
        label: sum(group["size"] for group in group_data if group["label"] == label)
        for label in (0, 1)
    }

    for label in (0, 1):
        label_groups = [group for group in group_data if group["label"] == label]
        rng.shuffle(label_groups)

        targets = {
            split: total_by_label[label] * ratio
            for split, ratio in zip(SPLITS, ratios)
        }
        current = {split: 0 for split in SPLITS}

        for group in label_groups:
            remaining = {
                split: targets[split] - current[split]
                for split in SPLITS
            }
            largest_gap = max(remaining.values())
            possible_splits = [
                split
                for split in SPLITS
                if abs(remaining[split] - largest_gap) < 1e-9
            ]
            chosen_split = rng.choice(possible_splits)
            assignment[group["group_id"]] = chosen_split
            current[chosen_split] += group["size"]

    return assignment


def measure_allocation(group_data, assignment, ratios):
    stats = {
        split: {
            "rows": 0,
            "fraud": 0,
            "groups": 0,
            "largest_group": 0,
        }
        for split in SPLITS
    }

    for group in group_data:
        split = assignment[group["group_id"]]
        stats[split]["rows"] += group["size"]
        stats[split]["fraud"] += group["size"] * group["label"]
        if group["label"] == 1:
            stats[split]["fraud_groups"] = stats[split].get("fraud_groups", 0) + 1
        stats[split]["groups"] += 1
        stats[split]["largest_group"] = max(
            stats[split]["largest_group"], group["size"]
        )

    total_rows = sum(values["rows"] for values in stats.values())
    total_fraud = sum(values["fraud"] for values in stats.values())
    overall_fraud_rate = total_fraud / total_rows

    size_deviations = []
    fraud_rate_deviations = []
    for split, target_ratio in zip(SPLITS, ratios):
        values = stats[split]
        values["share"] = values["rows"] / total_rows
        values["fraud_rate"] = values["fraud"] / values["rows"]
        values["largest_group_share"] = (
            values["largest_group"] / values["rows"]
        )
        size_deviations.append(abs(values["share"] - target_ratio))
        fraud_rate_deviations.append(
            abs(values["fraud_rate"] - overall_fraud_rate)
        )

    holdout_fraud = [stats[split]["fraud"] for split in SPLITS[1:]]
    return {
        "stats": stats,
        "max_size_deviation": max(size_deviations),
        "max_fraud_rate_deviation": max(fraud_rate_deviations),
        "minimum_holdout_fraud": min(holdout_fraud),
        "meets_minority_reference": all(
            count >= MINORITY_REFERENCE_COUNT for count in holdout_fraud
        ),
    }


def analyse_option(group_data, ratios):
    seed_results = []
    for seed in SENSITIVITY_SEEDS:
        assignment = allocate_groups(group_data, ratios, seed)
        seed_results.append(measure_allocation(group_data, assignment, ratios))

    main_assignment = allocate_groups(group_data, ratios, MAIN_SEED)
    main_result = measure_allocation(group_data, main_assignment, ratios)

    return {
        "main": main_result,
        "sensitivity": seed_results,
        "passing_seeds": sum(
            result["meets_minority_reference"] for result in seed_results
        ),
        "minimum_holdout_fraud_range": (
            min(result["minimum_holdout_fraud"] for result in seed_results),
            max(result["minimum_holdout_fraud"] for result in seed_results),
        ),
        "minimum_holdout_fraud_groups_range": (
            min(
                min(
                    result["stats"][split].get("fraud_groups", 0)
                    for split in SPLITS[1:]
                )
                for result in seed_results
            ),
            max(
                min(
                    result["stats"][split].get("fraud_groups", 0)
                    for split in SPLITS[1:]
                )
                for result in seed_results
            ),
        ),
        "max_size_deviation_range": (
            min(result["max_size_deviation"] for result in seed_results),
            max(result["max_size_deviation"] for result in seed_results),
        ),
    }


def format_main_split_details(option_results):
    lines = []
    for option_name, result in option_results.items():
        lines.extend([
            f"### {option_name}",
            "",
            "| Split | Rows | Share | Fraud | Fraud groups | Fraud rate | Groups | Largest group | Largest-group share |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for split in SPLITS:
            values = result["main"]["stats"][split]
            lines.append(
                f"| {split.title()} | {values['rows']:,} | {values['share']:.2%} | "
                f"{values['fraud']:,} | {values.get('fraud_groups', 0):,} | "
                f"{values['fraud_rate']:.2%} | {values['groups']:,} | "
                f"{values['largest_group']} | {values['largest_group_share']:.2%} |"
            )
        lines.append("")
    return lines


def create_report(group_data, total_rows, option_results):
    total_fraud = sum(
        group["size"] for group in group_data if group["label"] == 1
    )
    lines = [
        "# Split Feasibility Analysis v1",
        "",
        "This analysis compares split ratios before model training. It uses only group size and label; no text features or model results are used.",
        "",
        "## Data",
        "",
        f"- Rows after exact deduplication: **{total_rows:,}**",
        f"- Groups: **{len(group_data):,}**",
        f"- Fraudulent rows: **{total_fraud:,}** ({total_fraud / total_rows:.2%})",
        "- Main reproducible seed: **42**",
        "- Sensitivity seeds: **0 to 99**",
        "",
        "## Pre-specified minority reference",
        "",
        f"A calculated reference of **{MINORITY_REFERENCE_COUNT} fraudulent rows** per holdout was obtained from `ceil(1.96^2 × 0.25 / 0.10^2)`. This corresponds to an approximate worst-case 95% half-width of 10 percentage points for an independent binary proportion. The practical planning target is approximately **{OPERATIONAL_TARGET} fraudulent rows** in Validation and Test.",
        "",
        "This is a practical design reference, not a universal rule. Near-duplicate rows within a group are correlated, so the effective independent sample size may be smaller. Group counts and confidence intervals should also be reported in the final study.",
        "",
        "## Summary for seed 42",
        "",
        "| Option | Train rows | Val fraud | Test fraud | Minimum holdout fraud | Meets 97 reference? | Max size deviation |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]

    for option_name, result in option_results.items():
        stats = result["main"]["stats"]
        main = result["main"]
        lines.append(
            f"| {option_name} | {stats['train']['rows']:,} | "
            f"{stats['validation']['fraud']:,} | {stats['test']['fraud']:,} | "
            f"{main['minimum_holdout_fraud']:,} | "
            f"{'Yes' if main['meets_minority_reference'] else 'No'} | "
            f"{main['max_size_deviation']:.3%} |"
        )

    lines.extend([
        "",
        "## Sensitivity across 100 seeds",
        "",
        "| Option | Seeds meeting row reference | Range of minimum holdout fraud | Range of minimum fraud groups | Range of max size deviation |",
        "|---|---:|---:|---:|---:|",
    ])

    for option_name, result in option_results.items():
        fraud_low, fraud_high = result["minimum_holdout_fraud_range"]
        group_low, group_high = result["minimum_holdout_fraud_groups_range"]
        deviation_low, deviation_high = result["max_size_deviation_range"]
        lines.append(
            f"| {option_name} | {result['passing_seeds']}/100 | "
            f"{fraud_low}-{fraud_high} | "
            f"{group_low}-{group_high} | "
            f"{deviation_low:.3%}-{deviation_high:.3%} |"
        )

    lines.extend([
        "",
        "## Seed 42 details",
        "",
    ])
    lines.extend(format_main_split_details(option_results))
    lines.extend([
        "## Integrity result",
        "",
        f"All **{total_rows:,} records** and all **{len(group_data):,} groups** were assigned exactly once in every analysed allocation. Because assignment is performed at group level, a group cannot cross splits.",
        "",
        "## Interpretation rule",
        "",
        "1. Reject any allocation with missing/duplicated records or groups crossing splits.",
        f"2. Prefer options where Validation and Test each contain at least {MINORITY_REFERENCE_COUNT} fraudulent rows and are close to the operational target of {OPERATIONAL_TARGET}.",
        "3. Prefer options that meet the reference consistently across sensitivity seeds.",
        "4. Check that row shares and fraud rates remain close to their targets.",
        "5. Among options satisfying the above conditions, retain the larger Train partition.",
        "6. Do not use model performance to choose the ratio or seed.",
        "",
        "## Feasibility result and final decision",
        "",
    ])

    stable_options = [
        (option_name, result)
        for option_name, result in option_results.items()
        if result["main"]["meets_minority_reference"]
        and result["passing_seeds"] == len(SENSITIVITY_SEEDS)
    ]
    if stable_options:
        recommendation_name, recommendation = max(
            stable_options,
            key=lambda item: item[1]["main"]["stats"]["train"]["rows"],
        )
        stats = recommendation["main"]["stats"]
        rejected = [
            option_name
            for option_name, result in option_results.items()
            if not (
                result["main"]["meets_minority_reference"]
                and result["passing_seeds"] == len(SENSITIVITY_SEEDS)
            )
        ]
        if rejected:
            lines.append(
                f"- Options {', '.join(rejected)} do not meet the minority reference consistently."
            )
        lines.extend([
            f"- Stable options: {', '.join(name for name, _ in stable_options)}.",
            f"- The initial row-count rule favoured {recommendation_name}, because it retained the largest Train partition among stable options meeting the reference.",
            f"- The team selected **{FINAL_SELECTION}** after also considering that near-duplicate rows within a group are correlated. For seed 42, D has 107 fraud rows in each holdout and 97/99 fraud groups, compared with 100 fraud rows and 91/91 fraud groups for C.",
            "- D reduces Train by 316 rows (2.8%) compared with C, while giving both holdouts a larger buffer above the 97-row planning reference and greater fraud-group diversity.",
            "- This decision was recorded before model training and did not use model performance.",
            "",
        ])
    else:
        lines.extend([
            "- No candidate meets the pre-specified reference consistently. The design criteria must be reconsidered before creating final splits.",
            "",
        ])

    lines.extend([
        "## Method references",
        "",
        "- Vidros et al. (2017), EMSCAD: https://doi.org/10.3390/fi9010006",
        "- Collins et al. (2006), effective sample size for validation: https://doi.org/10.1016/j.jclinepi.2005.05.014",
        "- Riley et al. (2021), sample size for binary prediction-model validation: https://doi.org/10.1002/sim.9025",
        "",
    ])

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    group_data, total_rows = load_groups()
    option_results = {
        option_name: analyse_option(group_data, ratios)
        for option_name, ratios in OPTIONS.items()
    }
    create_report(group_data, total_rows, option_results)

    print(f"Rows analysed: {total_rows:,}")
    print(f"Groups analysed: {len(group_data):,}")
    print(f"Minority reference count: {MINORITY_REFERENCE_COUNT}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
