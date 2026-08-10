"""BERT + class-weight imbalance handling (default BERT hyperparameters).

VS Code / Cursor: open this file and click Run Python File.

- Uses sprint3/data/splits train/validation/test
- Default BertFinetuneConfig (lr=2e-5, epochs=5, max_length=384, ...)
- Train-only inverse-frequency class weights; no SMOTE / no random oversample
- Writes weights under ../weight/class_weight/
- Updates ../result/comparison_test.csv (test Accuracy / Fraud P / R / F1 only)
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import default_bert_config, load_splits, run_experiment


def main() -> int:
    splits = load_splits()
    cfg = default_bert_config(
        model_label="BERT + Class Weight",
        use_class_weights=True,
        class_weight_transform="none",  # classic inverse-frequency class weight
        class_weight_max=100.0,
        fraud_oversample_factor=1.0,
        optimize_threshold=True,
        threshold_selection_mode="max_fraud_f1",
        min_fraud_recall_for_threshold=None,
    )
    run_experiment(strategy_name="class_weight", cfg=cfg, splits=splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
