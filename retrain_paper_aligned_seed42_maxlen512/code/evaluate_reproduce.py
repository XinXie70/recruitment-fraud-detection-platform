"""Evaluate the saved paper-aligned checkpoint on the seed42 test split.

Writes metrics into ../results_eval and compares against the reported 0.9003.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

import config as cfg

# Route artifacts to results_eval for this script only.
cfg.RESULTS_DIR = cfg.PROJECT_ROOT / "results_eval"
cfg.FIGURES_DIR = cfg.RESULTS_DIR / "figures"
cfg.LOGS_DIR = cfg.RESULTS_DIR / "logs"
cfg.ensure_directories()

from dataset import dataframe_to_text_dataset, load_split  # noqa: E402
from metrics import (  # noqa: E402
    binary_metrics,
    build_error_analysis,
    build_predictions_frame,
    plot_confusion_matrix,
    plot_roc_pr,
)
from model import BertForFraudClassification  # noqa: E402
from train_bert import collate_batch, evaluate_loader  # noqa: E402
from utils import get_device, load_json, save_json, set_seed, setup_logging  # noqa: E402


TARGET_FRAUD_F1 = 0.9003021148036254


def main() -> None:
    logger = setup_logging("eval_reproduce.log")
    set_seed(42)
    device = get_device(allow_cpu=False, logger=logger)

    ckpt = cfg.REFERENCE_CHECKPOINT
    if not ckpt.exists():
        raise FileNotFoundError(f"Reference checkpoint missing: {ckpt}")

    thr_path = ckpt / "threshold.json"
    threshold = float(load_json(thr_path)["threshold"]) if thr_path.exists() else 0.99
    max_length = 256
    eval_batch_size = 32

    logger.info("Checkpoint: %s", ckpt)
    logger.info("threshold=%.4f max_length=%s", threshold, max_length)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    # Load fine-tuned weights directly to avoid a fresh randomly-init classifier.
    from transformers import AutoModelForSequenceClassification

    model = BertForFraudClassification.__new__(BertForFraudClassification)
    torch.nn.Module.__init__(model)
    model.model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model = model.to(device)
    model.eval()

    test_df = load_split("test")
    ds = dataframe_to_text_dataset(test_df, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(
        ds,
        batch_size=eval_batch_size,
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
        model_name="BERT:reproduce_eval",
        imbalance_strategy="checkpoint",
        row_indices=idxs,
    )
    preds.to_csv(cfg.RESULTS_DIR / "predictions_eval.csv", index=False)
    errors = build_error_analysis(preds, test_df["model_text"].tolist())
    errors.to_csv(cfg.RESULTS_DIR / "error_analysis_eval.csv", index=False)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        cfg.FIGURES_DIR / "confusion_matrix_eval.png",
        title="Reproduce eval Confusion Matrix",
    )
    plot_roc_pr(
        y_true,
        y_prob,
        cfg.FIGURES_DIR / "roc_curve_eval.png",
        cfg.FIGURES_DIR / "precision_recall_curve_eval.png",
    )

    delta = float(metrics["fraud_f1"]) - TARGET_FRAUD_F1
    payload = {
        "mode": "evaluate_saved_checkpoint",
        "checkpoint": str(ckpt),
        "threshold": threshold,
        "max_length": max_length,
        "n_test": int(len(test_df)),
        "target_fraud_f1": TARGET_FRAUD_F1,
        "reproduced_fraud_f1": float(metrics["fraud_f1"]),
        "delta_fraud_f1": delta,
        "match_within_1e-6": abs(delta) < 1e-6,
        "test_metrics": metrics,
        "test_metrics_threshold_0_5": metrics_05,
    }
    out = cfg.RESULTS_DIR / "reproduce_eval_metrics.json"
    save_json(payload, out)

    logger.info(
        "Reproduced test fraud_f1=%.6f (target=%.6f, delta=%+.6f)",
        metrics["fraud_f1"],
        TARGET_FRAUD_F1,
        delta,
    )
    logger.info(
        "fraud_precision=%.4f fraud_recall=%.4f pr_auc=%.4f",
        metrics["fraud_precision"],
        metrics["fraud_recall"],
        metrics["pr_auc"],
    )
    logger.info("Wrote %s", out)
    print(json.dumps({k: payload[k] for k in [
        "reproduced_fraud_f1", "target_fraud_f1", "delta_fraud_f1", "match_within_1e-6"
    ]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
