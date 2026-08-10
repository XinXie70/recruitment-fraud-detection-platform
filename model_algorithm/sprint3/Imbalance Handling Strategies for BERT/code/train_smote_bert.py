"""BERT + SMOTE imbalance handling (default BERT hyperparameters).

VS Code / Cursor: open this file and click Run Python File.

- Uses sprint3/data/splits train/validation/test
- Default BertFinetuneConfig (lr=2e-5, epochs=5, max_length=384, ...)
- Train-only SMOTE in TF-IDF space, mapped back to nearest original texts
- No class-weight loss; validation/test are never augmented
- Writes weights under ../weight/smote/
- Updates ../result/comparison_test.csv (test Accuracy / Fraud P / R / F1 only)
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import (
    apply_smote_text_augmentation,
    default_bert_config,
    load_splits,
    run_experiment,
)


def main() -> int:
    splits = load_splits()
    train_aug = apply_smote_text_augmentation(
        splits["train"],
        text_column="model_text",
        label_column="label",
        seed=42,
    )
    splits = {
        "train": train_aug,
        "validation": splits["validation"],
        "test": splits["test"],
    }
    cfg = default_bert_config(
        model_label="BERT + SMOTE",
        use_class_weights=False,
        fraud_oversample_factor=1.0,
        optimize_threshold=True,
        threshold_selection_mode="max_fraud_f1",
        min_fraud_recall_for_threshold=None,
    )
    run_experiment(strategy_name="smote", cfg=cfg, splits=splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
