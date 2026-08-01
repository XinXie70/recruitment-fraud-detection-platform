#!/usr/bin/env python
"""Paper-protocol vs Project-protocol BERT comparison.

Paper protocol (Jonathan et al., ICITCOM 2025 style):
  - EMSCAD full rows (Condition A input, 17,880)
  - 80/20 stratified train/test, then 10% of train -> validation
  - seed = 42
  - bert-base-uncased, max_length=256, lr=5e-5, batch=16, epochs=3
  - inverse-frequency class weights (no oversample, no sqrt_clip)
  - structured_combined_text with per-section word caps
  - threshold: max fraud F1 on validation (no min-recall constraint)

Project protocol (already trained):
  - Condition B exact-dedup, 70/15/15, seed 42
  - bert_cw_improved (384, tagged, sqrt_clip, oversample×3, min_recall=0.85)

Usage:
  E:\\ml\\venv\\Scripts\\python.exe run_bert_paper_vs_project.py
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    BertFinetuneConfig,
    ensure_directories,
)
from metrics import binary_metrics
from train_bert import run_training
from utils import save_json, setup_logging


EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "bert_paper_vs_project"
PAPER_DIR = EXPERIMENT_DIR / "paper_protocol_emscad_seed42"
PROJECT_B_DIR = (
    PROJECT_ROOT / "experiments" / "bert_cw_improved_condition_b_seed42"
)
INPUT_A = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emscad_condition_a_no_dedup_input_v1.csv.gz"
)
ASSIGNMENT_FILE = PAPER_DIR / "paper_80_20_seed42_assignments.csv.gz"
WEIGHTS_DIR = PAPER_DIR / "weights"
RESULTS_DIR = PAPER_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SEED = 42
RUN_NAME = "bert_paper_protocol"

# Approximate paper structured_combined_text word budgets.
SECTION_WORD_CAPS = {
    "title": 20,
    "company_profile": 80,
    "description": 300,
    "requirements": 120,
    "benefits": 80,
}
SECTION_ORDER = list(SECTION_WORD_CAPS.keys())


def _cap_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def build_structured_combined_text(combined: str) -> str:
    """Best-effort paper-style structured text from newline-joined combined_text."""
    if combined is None or (isinstance(combined, float) and pd.isna(combined)):
        return ""
    lines = [ln.strip() for ln in str(combined).split("\n") if ln.strip()]
    if not lines:
        return ""

    parts: List[str] = []
    # Map lines onto the five EMSCAD text fields used by the project pipeline.
    if len(lines) == 1:
        parts.append(f"[TITLE] {_cap_words(lines[0], SECTION_WORD_CAPS['title'])}")
        return "\n".join(parts)

    parts.append(f"[TITLE] {_cap_words(lines[0], SECTION_WORD_CAPS['title'])}")
    rest = lines[1:]
    if len(rest) == 1:
        parts.append(
            f"[DESCRIPTION] {_cap_words(rest[0], SECTION_WORD_CAPS['description'])}"
        )
        return "\n".join(parts)

    parts.append(
        f"[COMPANY PROFILE] {_cap_words(rest[0], SECTION_WORD_CAPS['company_profile'])}"
    )
    body = rest[1:]
    if not body:
        return "\n".join(parts)

    n = len(body)
    if n == 1:
        parts.append(
            f"[DESCRIPTION] {_cap_words(body[0], SECTION_WORD_CAPS['description'])}"
        )
        return "\n".join(parts)

    c1 = max(1, n // 3)
    c2 = max(c1 + 1, (2 * n) // 3)
    desc = " ".join(body[:c1])
    reqs = " ".join(body[c1:c2])
    bens = " ".join(body[c2:])
    parts.append(f"[DESCRIPTION] {_cap_words(desc, SECTION_WORD_CAPS['description'])}")
    if reqs.strip():
        parts.append(
            f"[REQUIREMENTS] {_cap_words(reqs, SECTION_WORD_CAPS['requirements'])}"
        )
    if bens.strip():
        parts.append(f"[BENEFITS] {_cap_words(bens, SECTION_WORD_CAPS['benefits'])}")
    return "\n".join(parts)


def ensure_dirs() -> None:
    for p in (EXPERIMENT_DIR, PAPER_DIR, WEIGHTS_DIR, RESULTS_DIR, FIGURES_DIR):
        p.mkdir(parents=True, exist_ok=True)


def make_paper_assignment(data: pd.DataFrame) -> pd.DataFrame:
    """80/20 stratified, then 10% of the train pool -> validation (paper style)."""
    labels = data[LABEL_COLUMN]
    idx = np.arange(len(data))
    train_pool, test_idx = train_test_split(
        idx,
        test_size=0.20,
        stratify=labels,
        random_state=SEED,
    )
    train_idx, val_idx = train_test_split(
        train_pool,
        test_size=0.10,
        stratify=labels.iloc[train_pool],
        random_state=SEED,
    )
    split = pd.Series("", index=data.index)
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "validation"
    split.iloc[test_idx] = "test"
    assign = pd.DataFrame(
        {
            "record_id": data[ID_COLUMN].astype(str),
            "seed": SEED,
            "split": split,
        }
    )
    assign.to_csv(ASSIGNMENT_FILE, index=False, compression="gzip")
    return assign


def prepare_paper_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    df["model_text"] = df["combined_text"].map(build_structured_combined_text)
    empty = (df["model_text"].str.strip() == "").sum()
    if empty:
        # Fallback to raw combined_text if structuring failed.
        mask = df["model_text"].str.strip() == ""
        df.loc[mask, "model_text"] = df.loc[mask, "combined_text"].fillna("").astype(str)
    df["title"] = df["model_text"].map(lambda t: t.split("\n", 1)[0][:120])
    df["company"] = ""
    return df


def build_splits(data: pd.DataFrame, assignments: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    selected = assignments[assignments["seed"] == SEED].copy()
    selected[ID_COLUMN] = selected[ID_COLUMN].astype(str)
    merged = selected.merge(data, on=ID_COLUMN, how="left", validate="one_to_one")
    if merged["model_text"].isna().any():
        raise ValueError("Missing model_text after merge")
    splits = {}
    for name in ("train", "validation", "test"):
        part = merged[merged["split"] == name].copy().reset_index(drop=True)
        part["split"] = name
        part["row_index"] = np.arange(len(part), dtype=np.int64)
        splits[name] = part
    return splits


def extract_metrics_from_project() -> Dict:
    summary_csv = PROJECT_B_DIR / "results" / "val_test_metrics_summary.csv"
    meta = PROJECT_B_DIR / "results" / "experiment_meta.json"
    if not summary_csv.exists():
        return {}
    df = pd.read_csv(summary_csv)
    out = {"source": str(summary_csv), "rows": df.to_dict(orient="records")}
    if meta.exists():
        out["meta"] = json.loads(meta.read_text(encoding="utf-8"))
    return out


def write_comparison(paper_payload: Dict, project_info: Dict) -> Path:
    paper_val = paper_payload["validation_metrics_optimized_threshold"]
    paper_test = paper_payload["test_metrics"]

    rows = [
        {
            "protocol": "Paper-aligned",
            "data": "EMSCAD full (Cond A input)",
            "split": "80/20 + 10%val-of-train",
            "seed": SEED,
            "max_length": 256,
            "lr": 5e-5,
            "epochs": 3,
            "imbalance": "class_weight(raw)",
            "eval_split": "validation",
            "threshold": paper_payload["threshold"],
            "fraud_precision": paper_val["fraud_precision"],
            "fraud_recall": paper_val["fraud_recall"],
            "fraud_f1": paper_val["fraud_f1"],
            "pr_auc": paper_val["pr_auc"],
            "roc_auc": paper_val["roc_auc"],
            "accuracy": paper_val["accuracy"],
        },
        {
            "protocol": "Paper-aligned",
            "data": "EMSCAD full (Cond A input)",
            "split": "80/20 + 10%val-of-train",
            "seed": SEED,
            "max_length": 256,
            "lr": 5e-5,
            "epochs": 3,
            "imbalance": "class_weight(raw)",
            "eval_split": "test",
            "threshold": paper_payload["threshold"],
            "fraud_precision": paper_test["fraud_precision"],
            "fraud_recall": paper_test["fraud_recall"],
            "fraud_f1": paper_test["fraud_f1"],
            "pr_auc": paper_test["pr_auc"],
            "roc_auc": paper_test["roc_auc"],
            "accuracy": paper_test["accuracy"],
        },
    ]

    # Paper reported numbers (reference only).
    rows.append(
        {
            "protocol": "Paper reported (reference)",
            "data": "EMSCAD (paper)",
            "split": "80/20 stratified",
            "seed": 42,
            "max_length": 256,
            "lr": 5e-5,
            "epochs": 3,
            "imbalance": "class_weight",
            "eval_split": "test",
            "threshold": None,
            "fraud_precision": 0.9130,
            "fraud_recall": 0.8497,
            "fraud_f1": 0.8802,
            "pr_auc": None,
            "roc_auc": 0.9909,
            "accuracy": 0.9888,
        }
    )

    for r in project_info.get("rows", []):
        split_name = str(r.get("split", ""))
        rows.append(
            {
                "protocol": "Project improved CW",
                "data": "Condition B exact-dedup",
                "split": "70/15/15",
                "seed": 42,
                "max_length": 384,
                "lr": 2e-5,
                "epochs": 5,
                "imbalance": "oversample×3 + sqrt_clip",
                "eval_split": split_name,
                "threshold": r.get("threshold"),
                "fraud_precision": r.get("fraud_precision"),
                "fraud_recall": r.get("fraud_recall"),
                "fraud_f1": r.get("fraud_f1"),
                "pr_auc": r.get("pr_auc"),
                "roc_auc": r.get("roc_auc"),
                "accuracy": r.get("accuracy"),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = EXPERIMENT_DIR / "comparison_summary.csv"
    df.to_csv(out_csv, index=False)

    md_lines = [
        "# BERT Paper Protocol vs Project Protocol",
        "",
        "## Design",
        "",
        "| Arm | Data | Split | Model recipe |",
        "|---|---|---|---|",
        "| **Paper-aligned** | EMSCAD full (17,880) | 80/20 + 10% of train as Val, seed 42 | max_len=256, lr=5e-5, batch=16, epochs=3, raw class weight |",
        "| **Project improved** | Condition B (~15,807) | 70/15/15, seed 42 | max_len=384, lr=2e-5, oversample×3 + sqrt_clip, min_recall≥0.85 |",
        "| **Paper reported** | EMSCAD (authors) | 80/20, seed 42 | numbers from the paper (reference) |",
        "",
        "## Results",
        "",
        "| Protocol | Eval | Fraud F1 | Precision | Recall | PR-AUC | ROC-AUC | Acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        def fmt(x):
            return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.4f}"

        md_lines.append(
            f"| {r['protocol']} | {r['eval_split']} | {fmt(r['fraud_f1'])} | "
            f"{fmt(r['fraud_precision'])} | {fmt(r['fraud_recall'])} | "
            f"{fmt(r['pr_auc'])} | {fmt(r['roc_auc'])} | {fmt(r['accuracy'])} |"
        )

    md_lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Paper-aligned and Project arms use **different data/split protocols**; gaps are expected.",
            "- If Paper-aligned Test F1 approaches the paper's 0.8802, the remaining gap to Project Holdout is largely **evaluation design**, not a broken BERT implementation.",
            "- Project Val F1 can exceed Paper Test F1 while Holdout is slightly lower — that is consistent with a stricter/different holdout.",
            "",
            f"- Paper artifacts: `{PAPER_DIR.as_posix()}`",
            f"- Project artifacts: `{PROJECT_B_DIR.as_posix()}`",
            "",
        ]
    )
    out_md = EXPERIMENT_DIR / "comparison_summary.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    return out_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip_train", action="store_true", help="Only rebuild comparison table")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    ensure_dirs()
    logger = setup_logging(str(RESULTS_DIR / "paper_protocol.log"))

    project_info = extract_metrics_from_project()
    if not project_info:
        logger.warning("Project B/seed42 metrics not found; comparison will be paper-only")

    if args.skip_train:
        metrics_path = RESULTS_DIR / f"bert_test_metrics_{RUN_NAME}.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        paper_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        out = write_comparison(paper_payload, project_info)
        logger.info("Comparison written to %s", out)
        return

    data = prepare_paper_frame(INPUT_A)
    if ASSIGNMENT_FILE.exists():
        assignments = pd.read_csv(ASSIGNMENT_FILE)
    else:
        assignments = make_paper_assignment(data)
        logger.info("Wrote paper assignment: %s", ASSIGNMENT_FILE)

    splits = build_splits(data, assignments)
    for name, frame in splits.items():
        logger.info(
            "Paper split %s: n=%d fraud=%d",
            name,
            len(frame),
            int((frame[LABEL_COLUMN] == 1).sum()),
        )

    cfg = BertFinetuneConfig(
        use_class_weights=True,
        class_weight_transform="none",  # raw inverse-frequency like paper
        class_weight_max=100.0,
        fraud_oversample_factor=1.0,
        min_fraud_recall_for_threshold=None,  # unconstrained max fraud F1
        threshold_step=0.01,
        num_train_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        random_seed=SEED,
        output_dir=str(WEIGHTS_DIR),
        run_name=RUN_NAME,
        write_canonical_aliases=False,
        model_label="BERT-paper-protocol",
        early_stopping_patience=2,
    )

    try:
        payload = run_training(
            cfg,
            splits=splits,
            results_dir=RESULTS_DIR,
            figures_dir=FIGURES_DIR,
            log_name="paper_protocol_train.log",
        )
        save_json(
            {
                "protocol": "paper_aligned",
                "seed": SEED,
                "run_name": RUN_NAME,
                "input_file": str(INPUT_A),
                "assignment_file": str(ASSIGNMENT_FILE),
                "split_counts": {
                    k: {
                        "n": len(v),
                        "fraud": int((v[LABEL_COLUMN] == 1).sum()),
                    }
                    for k, v in splits.items()
                },
                "config_notes": {
                    "max_length": args.max_length,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "batch_size": args.train_batch_size,
                    "class_weight": "raw inverse-frequency",
                    "text": "structured_combined_text (approx from combined_text)",
                },
                "test_fraud_f1": payload["test_metrics"]["fraud_f1"],
                "test_pr_auc": payload["test_metrics"]["pr_auc"],
                "threshold": payload.get("threshold"),
            },
            RESULTS_DIR / "experiment_meta.json",
        )
        out = write_comparison(payload, project_info)
        logger.info(
            "Paper protocol done. Test fraud_f1=%.4f pr_auc=%.4f | comparison=%s",
            payload["test_metrics"]["fraud_f1"],
            payload["test_metrics"]["pr_auc"],
            out,
        )
    except SystemExit:
        raise
    except Exception:
        logger.error("Paper protocol training failed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
