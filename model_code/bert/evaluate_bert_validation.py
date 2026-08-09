"""Run locked BERT checkpoint inference on validation.csv or test.csv.

Does not train or modify the model. Writes ensemble-ready predictions with the
shared contract columns under reports/models/bert/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import BERT_FINETUNED_DIR, PRETRAINED_MODEL_NAME, PROJECT_ROOT, ensure_directories
from dataset import dataframe_to_text_dataset, load_split
from metrics import binary_metrics
from model import BertForFraudClassification
from train_bert import collate_batch, evaluate_loader
from utils import get_device, load_json, save_json, set_seed, setup_logging


REPORT_DIR = PROJECT_ROOT / "reports" / "models" / "bert"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved BERT checkpoint on validation.csv or test.csv"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(BERT_FINETUNED_DIR / "bert_class_weighted" / "best"),
    )
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    split_name = args.split

    ensure_directories()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
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

    if args.model_name is not None:
        model_name = args.model_name
    elif thr_path.exists():
        model_name = str(load_json(thr_path).get("run_name", ckpt.parent.name))
    else:
        model_name = ckpt.parent.name

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = BertForFraudClassification(PRETRAINED_MODEL_NAME)
    model.model = type(model.model).from_pretrained(ckpt)
    model = model.to(device)

    validation_df = load_split(split_name)
    dataset = dataframe_to_text_dataset(validation_df, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda feats: collate_batch(feats, collator),
        num_workers=0,
    )

    _, y_true, y_prob, record_ids, _, _, _ = evaluate_loader(
        model, loader, device, criterion=None, use_amp=device.type == "cuda"
    )
    metrics = binary_metrics(y_true, y_prob, threshold=threshold)

    predictions = pd.DataFrame(
        {
            "record_id": record_ids,
            "model_name": model_name,
            "fraud_score": y_prob,
            "threshold": threshold,
            "prediction": (y_prob >= threshold).astype(int),
            "true_label": y_true,
        }
    )
    pred_filename = f"{split_name}_predictions.csv"
    metrics_filename = f"{split_name}_metrics.json"
    predictions.to_csv(REPORT_DIR / pred_filename, index=False)

    metrics_payload = {
        "checkpoint": str(ckpt),
        "model_name": model_name,
        "split": split_name,
        "threshold": threshold,
        f"{split_name}_metrics": metrics,
        "num_rows": int(len(predictions)),
    }
    save_json(metrics_payload, REPORT_DIR / metrics_filename)

    logger.info(
        "%s fraud_f1=%.4f fraud_recall=%.4f pr_auc=%.4f accuracy=%.4f",
        split_name.capitalize(),
        metrics["fraud_f1"],
        metrics["fraud_recall"],
        metrics["pr_auc"],
        metrics["accuracy"],
    )
    print(f"Wrote {len(predictions)} rows to {REPORT_DIR / pred_filename}")
    print(f"Model: {model_name}")
    print(f"Fraud F1: {metrics['fraud_f1']:.4f}")
    print(f"Fraud Recall: {metrics['fraud_recall']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
