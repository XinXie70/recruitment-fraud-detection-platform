"""Evaluate retrain checkpoint on test with fixed threshold 0.99."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import config as cfg

cfg.RESULTS_DIR = cfg.PROJECT_ROOT / "results"
cfg.FIGURES_DIR = cfg.RESULTS_DIR / "figures"
cfg.LOGS_DIR = cfg.RESULTS_DIR / "logs"
cfg.ensure_directories()

from dataset import dataframe_to_text_dataset, load_split
from metrics import binary_metrics, build_predictions_frame
from model import BertForFraudClassification
from train_bert import collate_batch, evaluate_loader
from utils import get_device, save_json, set_seed, setup_logging


def main() -> None:
    logger = setup_logging("eval_thr099.log")
    set_seed(42)
    device = get_device(allow_cpu=False, logger=logger)

    ckpt = cfg.PROJECT_ROOT / "weights" / "bert_paper_protocol" / "best"
    threshold = 0.99
    max_length = 256
    logger.info("ckpt=%s thr=%.2f max_length=%s", ckpt, threshold, max_length)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = BertForFraudClassification.__new__(BertForFraudClassification)
    torch.nn.Module.__init__(model)
    model.model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model = model.to(device).eval()

    test_df = load_split("test")
    ds = dataframe_to_text_dataset(test_df, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda feats: collate_batch(feats, collator),
        num_workers=0,
    )

    _, y_true, y_prob, ids, titles, companies, idxs = evaluate_loader(
        model, loader, device, criterion=None, use_amp=True
    )
    metrics = binary_metrics(y_true, y_prob, threshold=threshold)
    metrics_003 = binary_metrics(y_true, y_prob, threshold=0.03)

    payload = {
        "checkpoint": str(ckpt),
        "threshold_fixed": threshold,
        "n_test": int(len(test_df)),
        "test_metrics_thr_0_99": metrics,
        "test_metrics_thr_0_03_for_reference": metrics_003,
    }
    out = cfg.RESULTS_DIR / "test_metrics_threshold_0_99.json"
    save_json(payload, out)

    preds = build_predictions_frame(
        record_ids=ids,
        titles=titles,
        companies=companies,
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        model_name="BERT:retrain_thr0.99",
        imbalance_strategy="checkpoint",
        row_indices=idxs,
    )
    preds.to_csv(cfg.RESULTS_DIR / "predictions_threshold_0_99.csv", index=False)

    logger.info(
        "thr=0.99 fraud_f1=%.4f precision=%.4f recall=%.4f macro_f1=%.4f pr_auc=%.4f",
        metrics["fraud_f1"],
        metrics["fraud_precision"],
        metrics["fraud_recall"],
        metrics["macro_f1"],
        metrics["pr_auc"],
    )
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in [
                    "fraud_precision",
                    "fraud_recall",
                    "fraud_f1",
                    "macro_f1",
                    "pr_auc",
                    "accuracy",
                    "roc_auc",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
