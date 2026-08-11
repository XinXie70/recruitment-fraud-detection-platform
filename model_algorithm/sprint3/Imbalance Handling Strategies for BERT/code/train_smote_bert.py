#Train BERT with SMOTE for EMSCAD data
from __future__ import annotations
import sys
from pathlib import Path
CODE_PATH = Path(__file__).resolve().parent
if str(CODE_PATH) not in sys.path:
    sys.path.insert(0, str(CODE_PATH))
from common import apply_smote_text_augmentation, create_bert_config, read_splits, run_experiment
def main() -> int:
    splits = read_splits()
    training_data = apply_smote_text_augmentation(splits["train"], text_column="model_text", label_column="label", seed=42)
    splits = {
        "train": training_data,
        "validation": splits["validation"],
        "test": splits["test"],
    }
    config = create_bert_config(model_label="BERT + SMOTE", use_class_weights=False, fraud_oversample_factor=1.0, optimize_threshold=True, threshold_selection_mode="max_fraud_f1", min_fraud_recall_for_threshold=None)
    run_experiment(strategy_name="smote", config=config, splits=splits)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())