"""Save the fixed record assignments used by Conditions A, B and C."""

from pathlib import Path

import pandas as pd

from run_condition_a import INPUT_FILE as A_INPUT_FILE
from run_condition_a import SEEDS, split_indices
from run_condition_b import INPUT_FILE as BC_INPUT_FILE
from run_condition_c import build_group_data, group_split_indices


PROJECT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_DIR / "data/experiment_splits"
OUTPUT_FILES = {
    "A": OUTPUT_DIR
    / "condition_a_no_dedup_random_split_assignments_v1.csv.gz",
    "B": OUTPUT_DIR
    / "condition_b_exact_dedup_random_split_assignments_v1.csv.gz",
    "C": OUTPUT_DIR
    / "condition_c_exact_dedup_group_aware_split_assignments_v1.csv.gz",
}


def create_assignment_rows(data, splitter):
    rows = []
    for seed in SEEDS:
        train_indices, validation_indices, holdout_indices = splitter(seed)
        split_names = pd.Series("", index=data.index)
        split_names.iloc[train_indices] = "train"
        split_names.iloc[validation_indices] = "validation"
        split_names.iloc[holdout_indices] = "holdout"

        if (split_names == "").any():
            raise ValueError(f"Seed {seed} contains unassigned records")

        rows.append(pd.DataFrame({
            "record_id": data["record_id"],
            "seed": seed,
            "split": split_names,
        }))
    return pd.concat(rows, ignore_index=True)


def validate_assignments(data, assignments):
    expected_seeds = set(SEEDS)
    if set(assignments["seed"]) != expected_seeds:
        raise ValueError("Assignment file does not contain seeds 0 to 9")

    for seed in SEEDS:
        selected = assignments[assignments["seed"] == seed]
        if len(selected) != len(data):
            raise ValueError(f"Seed {seed} has the wrong number of rows")
        if selected["record_id"].duplicated().any():
            raise ValueError(f"Seed {seed} contains duplicate record IDs")
        if set(selected["record_id"]) != set(data["record_id"]):
            raise ValueError(f"Seed {seed} record IDs do not match the input")
        if set(selected["split"]) != {"train", "validation", "holdout"}:
            raise ValueError(f"Seed {seed} is missing a split")


def save_assignments(assignments, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(
        output_file,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


def main():
    condition_a = pd.read_csv(A_INPUT_FILE).reset_index(drop=True)
    conditions_bc = pd.read_csv(BC_INPUT_FILE).reset_index(drop=True)
    group_data = build_group_data(conditions_bc)

    assignments = {
        "A": create_assignment_rows(
            condition_a,
            lambda seed: split_indices(condition_a["label"], seed),
        ),
        "B": create_assignment_rows(
            conditions_bc,
            lambda seed: split_indices(conditions_bc["label"], seed),
        ),
        "C": create_assignment_rows(
            conditions_bc,
            lambda seed: group_split_indices(
                conditions_bc,
                group_data,
                seed,
            ),
        ),
    }

    for condition, data in [
        ("A", condition_a),
        ("B", conditions_bc),
        ("C", conditions_bc),
    ]:
        validate_assignments(data, assignments[condition])
        save_assignments(assignments[condition], OUTPUT_FILES[condition])
        print(
            f"Condition {condition}: {len(assignments[condition]):,} "
            f"assignment rows saved to {OUTPUT_FILES[condition]}"
        )


if __name__ == "__main__":
    main()
