"""Export BERT validation scores for the FP-gate ensemble.

Writes record_id,label,fraud_score to:
  ../results/bert_validation_predictions.csv

Also refreshes the ensemble copy when that directory exists:
  ../../ensemble_bert_fp_gate_lr_none_bigram_maxlen512/results/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import BERT_FINETUNED_DIR, PRETRAINED_MODEL_NAME, RESULTS_DIR, ensure_directories
from dataset import dataframe_to_text_dataset, load_split
from model import BertForFraudClassification
from train_bert import collate_batch, evaluate_loader
from utils import get_device, set_seed, setup_logging

DEFAULT_CHECKPOINT = BERT_FINETUNED_DIR / "bert_paper_protocol_maxlen512" / "best"
ENSEMBLE_COPY = (
    Path(__file__).resolve().parents[2]
    / "ensemble_bert_fp_gate_lr_none_bigram_maxlen512"
    / "results"
    / "bert_validation_predictions.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export BERT validation fraud scores for ensemble FP-gate"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    ensure_directories()
    logger = setup_logging("export_validation_predictions.log")
    set_seed(args.random_seed)
    device = get_device(allow_cpu=args.allow_cpu, logger=logger)

    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = BertForFraudClassification(PRETRAINED_MODEL_NAME)
    model.model = type(model.model).from_pretrained(ckpt)
    model = model.to(device)

    val_df = load_split("validation")
    ds = dataframe_to_text_dataset(val_df, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda feats: collate_batch(feats, collator),
        num_workers=0,
    )

    _, y_true, y_prob, ids, *_ = evaluate_loader(
        model, loader, device, criterion=None, use_amp=device.type == "cuda"
    )

    out = pd.DataFrame(
        {
            "record_id": list(ids),
            "label": y_true.astype(int),
            "fraud_score": y_prob.astype(float),
        }
    )
    out_path = RESULTS_DIR / "bert_validation_predictions.csv"
    out.to_csv(out_path, index=False)
    logger.info("Wrote %s (%s rows)", out_path, len(out))

    if ENSEMBLE_COPY.parent.exists():
        shutil.copy2(out_path, ENSEMBLE_COPY)
        logger.info("Also refreshed ensemble copy: %s", ENSEMBLE_COPY)


if __name__ == "__main__":
    main()
