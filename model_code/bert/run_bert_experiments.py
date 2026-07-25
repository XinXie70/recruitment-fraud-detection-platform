#!/usr/bin/env python
"""Run the retained BERT comparison on fixed splits.

  1) BERT fine-tune without class weights
  2) BERT fine-tune with class weights
"""

from __future__ import annotations

import argparse
import traceback

from config import BertFinetuneConfig, ensure_directories
from train_bert import run_training
from utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    ensure_directories()
    logger = setup_logging("training.log")

    try:
        logger.info("=== BERT fine-tune: no class weights ===")
        run_training(
            BertFinetuneConfig(
                use_class_weights=False,
                num_train_epochs=args.epochs,
                run_name="bert_no_class_weight",
            )
        )
        logger.info("=== BERT fine-tune: class weights ===")
        run_training(
            BertFinetuneConfig(
                use_class_weights=True,
                num_train_epochs=args.epochs,
                run_name="bert_class_weighted",
            )
        )
        logger.info("BERT None vs class-weight comparison finished.")
    except SystemExit:
        raise
    except Exception:
        logger.error("run_bert_experiments failed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
