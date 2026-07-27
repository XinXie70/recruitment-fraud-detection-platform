#!/usr/bin/env python
"""Train class-weighted BERT on split-leakage Conditions A / B / C.

Uses the shared experiment_splits assignment files (seeds 0–9 by default) so
BERT sees the same diagnostic Train / Validation / Holdout partitions as the
LR and Linear SVM runs.

Usage (PowerShell):
  . E:\\ml\\activate.ps1
  cd F:\\final-version-1\\capstone-project-26t2-9900-h09c-almond\\model_code\\bert
  python run_bert_split_leakage.py
  python run_bert_split_leakage.py --conditions A B C --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from config import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    RESULTS_DIR,
    WEIGHTS_DIR,
    BertFinetuneConfig,
    ensure_directories,
)
from preprocessing import (
    derive_company_preview,
    derive_title_preview,
    prepare_dataframe_text,
)
from train_bert import run_training
from utils import save_json, setup_logging


EXPERIMENT_NAME = "split_leakage"
HOLDING_SPLIT_NAME = "holdout"

CONDITIONS = {
    "A": {
        "label": "A: no dedup + stratified random split",
        "input_file": PROJECT_ROOT
        / "data"
        / "processed"
        / "emscad_condition_a_no_dedup_input_v1.csv.gz",
        "assignment_file": PROJECT_ROOT
        / "data"
        / "experiment_splits"
        / "condition_a_no_dedup_random_split_assignments_v1.csv.gz",
        "dir_name": "condition_a",
    },
    "B": {
        "label": "B: exact dedup + stratified random split",
        "input_file": PROJECT_ROOT
        / "data"
        / "processed"
        / "emscad_conditions_b_c_exact_dedup_grouped_input_v1.csv.gz",
        "assignment_file": PROJECT_ROOT
        / "data"
        / "experiment_splits"
        / "condition_b_exact_dedup_random_split_assignments_v1.csv.gz",
        "dir_name": "condition_b",
    },
    "C": {
        "label": "C: exact dedup + group-aware split",
        "input_file": PROJECT_ROOT
        / "data"
        / "processed"
        / "emscad_conditions_b_c_exact_dedup_grouped_input_v1.csv.gz",
        "assignment_file": PROJECT_ROOT
        / "data"
        / "experiment_splits"
        / "condition_c_exact_dedup_group_aware_split_assignments_v1.csv.gz",
        "dir_name": "condition_c",
    },
}

WEIGHTS_ROOT = WEIGHTS_DIR / EXPERIMENT_NAME
RESULTS_ROOT = RESULTS_DIR / EXPERIMENT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune class-weighted BERT on experiment_splits A/B/C"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["A", "B", "C"],
        choices=sorted(CONDITIONS),
        help="Which diagnostic conditions to run",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="Assignment seeds to train (default: 0..9)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a seed if its metrics JSON already exists",
    )
    return parser.parse_args()


def prepare_input_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{path} missing '{LABEL_COLUMN}'")
    if ID_COLUMN not in df.columns:
        raise ValueError(f"{path} missing '{ID_COLUMN}'")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    df = prepare_dataframe_text(df)
    if "title" not in df.columns:
        df["title"] = df["model_text"].map(derive_title_preview)
    if "company" not in df.columns:
        df["company"] = df["model_text"].map(derive_company_preview)
    return df


def finalise_split_frame(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["split"] = split_name
    out["row_index"] = np.arange(len(out), dtype=np.int64)
    return out


def build_splits_for_seed(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    selected = assignments[assignments["seed"] == seed].copy()
    if selected.empty:
        raise ValueError(f"No assignment rows for seed={seed}")
    if selected[ID_COLUMN].duplicated().any():
        raise ValueError(f"Duplicate record_id in assignments for seed={seed}")

    selected[ID_COLUMN] = selected[ID_COLUMN].astype(str)
    selected["split"] = selected["split"].astype(str).str.lower()
    # Diagnostic holdout is evaluated as the BERT "test" partition.
    selected.loc[selected["split"] == HOLDING_SPLIT_NAME, "split"] = "test"

    expected = {"train", "validation", "test"}
    found = set(selected["split"])
    if found != expected:
        raise ValueError(
            f"Seed {seed} splits={sorted(found)}; expected {sorted(expected)}"
        )

    merged = selected.merge(data, on=ID_COLUMN, how="left", validate="one_to_one")
    if merged["model_text"].isna().any() or merged[LABEL_COLUMN].isna().any():
        missing = int(merged["model_text"].isna().sum())
        raise ValueError(
            f"Seed {seed}: {missing} assignment IDs missing from input data"
        )

    splits = {}
    for name in ("train", "validation", "test"):
        part = merged[merged["split"] == name]
        splits[name] = finalise_split_frame(part, name)
    return splits


def checkpoint_dir(condition_key: str, seed: int) -> Path:
    meta = CONDITIONS[condition_key]
    return (
        WEIGHTS_ROOT
        / meta["dir_name"]
        / f"seed_{seed}"
        / "bert_class_weighted"
        / "best"
    )


def metrics_path_for(condition_key: str, seed: int) -> Path:
    meta = CONDITIONS[condition_key]
    return (
        RESULTS_ROOT
        / meta["dir_name"]
        / f"seed_{seed}"
        / "bert_test_metrics_bert_class_weighted.json"
    )


def flatten_payload(condition_key: str, seed: int, payload: Dict) -> Dict:
    test_metrics = payload["test_metrics"]
    return {
        "condition": condition_key,
        "condition_label": CONDITIONS[condition_key]["label"],
        "seed": seed,
        "best_epoch": payload.get("best_epoch"),
        "threshold": payload.get("threshold"),
        "training_time_sec": payload.get("training_time_sec"),
        "pr_auc": test_metrics["pr_auc"],
        "roc_auc": test_metrics["roc_auc"],
        "accuracy": test_metrics["accuracy"],
        "fraud_precision": test_metrics["fraud_precision"],
        "fraud_recall": test_metrics["fraud_recall"],
        "fraud_f1": test_metrics["fraud_f1"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics.get("weighted_f1"),
    }


def write_summary(rows: Sequence[Dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(list(rows))
    df.to_csv(RESULTS_ROOT / "bert_class_weighted_all_seeds.csv", index=False)

    metric_cols = [
        "pr_auc",
        "roc_auc",
        "accuracy",
        "fraud_precision",
        "fraud_recall",
        "fraud_f1",
        "macro_f1",
    ]
    summary = (
        df.groupby("condition", sort=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "condition" if c[0] == "condition" else f"{c[0]}_{c[1]}"
        for c in summary.columns.to_list()
    ]
    summary.to_csv(RESULTS_ROOT / "bert_class_weighted_summary.csv", index=False)

    lines = [
        "# Split Leakage Experiment: BERT (class-weighted)",
        "",
        "## Design",
        "",
        "- Model: **BERT fine-tune with train-only class weights**",
        "- Conditions: A / B / C from `data/experiment_splits/`",
        "- Holdout = diagnostic evaluation partition (mapped to BERT `test`)",
        f"- Seeds present in results: "
        f"**{', '.join(str(s) for s in sorted(df['seed'].unique()))}**",
        "",
        "## Mean diagnostic Holdout metrics",
        "",
        "| Condition | PR-AUC | ROC-AUC | Fraud F1 | Fraud Precision | Fraud Recall | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ["A", "B", "C"]:
        part = df[df["condition"] == condition]
        if part.empty:
            continue
        lines.append(
            "| {cond} | {pr:.4f} | {roc:.4f} | {f1:.4f} | {prec:.4f} | "
            "{rec:.4f} | {macro:.4f} |".format(
                cond=condition,
                pr=part["pr_auc"].mean(),
                roc=part["roc_auc"].mean(),
                f1=part["fraud_f1"].mean(),
                prec=part["fraud_precision"].mean(),
                rec=part["fraud_recall"].mean(),
                macro=part["macro_f1"].mean(),
            )
        )

    if set(df["condition"]) >= {"A", "B", "C"}:
        means = df.groupby("condition")[["pr_auc", "fraud_f1"]].mean()
        lines.extend(
            [
                "",
                "## Deltas",
                "",
                f"- B − A PR-AUC: **{means.loc['B', 'pr_auc'] - means.loc['A', 'pr_auc']:+.4f}**",
                f"- C − B PR-AUC: **{means.loc['C', 'pr_auc'] - means.loc['B', 'pr_auc']:+.4f}**",
                f"- B − A Fraud F1: **{means.loc['B', 'fraud_f1'] - means.loc['A', 'fraud_f1']:+.4f}**",
                f"- C − B Fraud F1: **{means.loc['C', 'fraud_f1'] - means.loc['B', 'fraud_f1']:+.4f}**",
            ]
        )

    lines.extend(
        [
            "",
            "## Artifact locations",
            "",
            f"- Weights: `{WEIGHTS_ROOT.as_posix()}/condition_*/seed_*/best/`",
            f"- Results: `{RESULTS_ROOT.as_posix()}/`",
            "",
        ]
    )
    (RESULTS_ROOT / "bert_class_weighted_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def train_one(
    condition_key: str,
    seed: int,
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict:
    meta = CONDITIONS[condition_key]
    results_dir = RESULTS_ROOT / meta["dir_name"] / f"seed_{seed}"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    splits = build_splits_for_seed(data, assignments, seed)
    # run_training saves to: output_dir / run_name / best
    # => .../split_leakage/condition_x/seed_k/bert_class_weighted/best
    cfg = BertFinetuneConfig(
        use_class_weights=True,
        num_train_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        random_seed=seed,
        output_dir=str(WEIGHTS_ROOT / meta["dir_name"] / f"seed_{seed}"),
        run_name="bert_class_weighted",
        write_canonical_aliases=False,
        model_label="BERT",
    )

    payload = run_training(
        cfg,
        splits=splits,
        results_dir=results_dir,
        figures_dir=figures_dir,
        log_name=f"split_leakage_{meta['dir_name']}_seed_{seed}.log",
    )
    save_json(
        {
            "condition": condition_key,
            "condition_label": meta["label"],
            "seed": seed,
            "assignment_file": str(meta["assignment_file"]),
            "input_file": str(meta["input_file"]),
            "weights_dir": str(checkpoint_dir(condition_key, seed)),
            "results_dir": str(results_dir),
            "sample_counts": {
                name: {
                    "n": len(frame),
                    "fraud": int((frame[LABEL_COLUMN] == 1).sum()),
                }
                for name, frame in splits.items()
            },
        },
        results_dir / "experiment_meta.json",
    )
    return payload


def main() -> None:
    args = parse_args()
    ensure_directories()
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("split_leakage_bert.log")

    conditions = [c.upper() for c in args.conditions]
    seeds = sorted(set(args.seeds))
    logger.info(
        "Starting BERT class-weighted split-leakage runs: conditions=%s seeds=%s",
        conditions,
        seeds,
    )

    cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    for key in conditions:
        meta = CONDITIONS[key]
        if not meta["input_file"].exists():
            raise FileNotFoundError(meta["input_file"])
        if not meta["assignment_file"].exists():
            raise FileNotFoundError(meta["assignment_file"])
        logger.info("Loading condition %s input + assignments", key)
        data = prepare_input_frame(meta["input_file"])
        assignments = pd.read_csv(meta["assignment_file"])
        assignments["seed"] = assignments["seed"].astype(int)
        assignments[ID_COLUMN] = assignments[ID_COLUMN].astype(str)
        cache[key] = {"data": data, "assignments": assignments}

    all_rows: List[Dict] = []
    for key in conditions:
        for seed in seeds:
            path = metrics_path_for(key, seed)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                all_rows.append(flatten_payload(key, seed, payload))

    try:
        for key in conditions:
            for seed in seeds:
                out_metrics = metrics_path_for(key, seed)
                if args.skip_existing and out_metrics.exists():
                    logger.info(
                        "Skip existing condition=%s seed=%s (%s)",
                        key,
                        seed,
                        out_metrics,
                    )
                    continue

                logger.info("=== Train condition %s seed %s ===", key, seed)
                payload = train_one(
                    key,
                    seed,
                    cache[key]["data"],
                    cache[key]["assignments"],
                    args,
                )
                all_rows = [
                    row
                    for row in all_rows
                    if not (row["condition"] == key and row["seed"] == seed)
                ]
                all_rows.append(flatten_payload(key, seed, payload))
                write_summary(all_rows)
                logger.info(
                    "Finished condition=%s seed=%s fraud_f1=%.4f pr_auc=%.4f",
                    key,
                    seed,
                    payload["test_metrics"]["fraud_f1"],
                    payload["test_metrics"]["pr_auc"],
                )

        write_summary(all_rows)
        logger.info("All requested BERT split-leakage runs finished.")
        logger.info("Weights: %s", WEIGHTS_ROOT)
        logger.info("Results: %s", RESULTS_ROOT)
    except SystemExit:
        raise
    except Exception:
        logger.error("split-leakage BERT training failed:\n%s", traceback.format_exc())
        write_summary(all_rows)
        raise


if __name__ == "__main__":
    main()
