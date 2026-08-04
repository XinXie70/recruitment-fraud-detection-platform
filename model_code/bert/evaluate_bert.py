"""Evaluate a saved BERT fine-tuned checkpoint on the fixed test split.

Does not tune hyperparameters or thresholds on test. Threshold is loaded from
the checkpoint's validation-selected threshold.json when available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import BERT_FINETUNED_DIR, FIGURES_DIR, PRETRAINED_MODEL_NAME, RESULTS_DIR, ensure_directories
from dataset import dataframe_to_text_dataset, load_split
from metrics import (
    binary_metrics,
    build_error_analysis,
    build_predictions_frame,
    plot_confusion_matrix,
    plot_roc_pr,
)
from model import BertForFraudClassification
from train_bert import collate_batch, evaluate_loader
from utils import get_device, load_json, save_json, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned BERT on test.csv")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(BERT_FINETUNED_DIR / "bert_class_weighted" / "best"),
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    ensure_directories()
    logger = setup_logging("training.log")
    set_seed(args.random_seed)
    device = get_device(allow_cpu=args.allow_cpu, logger=logger)

    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    thr_path = ckpt / "threshold.json"
    if args.threshold is not None:
        threshold = float(args.threshold)
    elif thr_path.exists():
        threshold = float(load_json(thr_path)["threshold"])
    else:
        threshold = 0.5
        logger.warning("No threshold.json found; using default 0.5")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = BertForFraudClassification(PRETRAINED_MODEL_NAME)
    model.model = type(model.model).from_pretrained(ckpt)
    model = model.to(device)

    test_df = load_split("test")
    ds = dataframe_to_text_dataset(test_df, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda feats: collate_batch(feats, collator),
        num_workers=0,
    )

    _, y_true, y_prob, ids, titles, companies, idxs = evaluate_loader(
        model, loader, device, criterion=None, use_amp=device.type == "cuda"
    )
    metrics = binary_metrics(y_true, y_prob, threshold=threshold)
    metrics_05 = binary_metrics(y_true, y_prob, threshold=0.5)

    preds = build_predictions_frame(
        record_ids=ids,
        titles=titles,
        companies=companies,
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        model_name=f"BERT:{ckpt.name}",
        imbalance_strategy="checkpoint",
        row_indices=idxs,
    )
    preds.to_csv(RESULTS_DIR / "predictions_eval.csv", index=False)
    errors = build_error_analysis(preds, test_df["model_text"].tolist())
    errors.to_csv(RESULTS_DIR / "error_analysis_eval.csv", index=False)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        FIGURES_DIR / "confusion_matrix_eval.png",
        title="BERT eval Confusion Matrix",
    )
    plot_roc_pr(
        y_true,
        y_prob,
        FIGURES_DIR / "roc_curve_eval.png",
        FIGURES_DIR / "precision_recall_curve_eval.png",
    )

    payload = {
        "checkpoint": str(ckpt),
        "threshold": threshold,
        "test_metrics": metrics,
        "test_metrics_threshold_0_5": metrics_05,
    }
    save_json(payload, RESULTS_DIR / "bert_eval_test_metrics.json")
    logger.info(
        "Test fraud_f1=%.4f fraud_recall=%.4f pr_auc=%.4f accuracy=%.4f",
        metrics["fraud_f1"],
        metrics["fraud_recall"],
        metrics["pr_auc"],
        metrics["accuracy"],
    )
    print(f"Fraud F1: {metrics['fraud_f1']:.4f}")
    print(f"Fraud Recall: {metrics['fraud_recall']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
