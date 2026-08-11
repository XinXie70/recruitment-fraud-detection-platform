#Train BERT with class weight for EMSCAD data
from __future__ import annotations
import sys
from pathlib import Path
CODE_PATH = Path(__file__).resolve().parent
if str(CODE_PATH) not in sys.path:
    sys.path.insert(0, str(CODE_PATH))
from common import create_bert_config, read_splits, run_experiment
def main() -> int:
    splits = read_splits()
    config = create_bert_config(model_label="BERT + Class Weight", use_class_weights=True, class_weight_transform="none", class_weight_max=100.0, fraud_oversample_factor=1.0, optimize_threshold=True, threshold_selection_mode="max_fraud_f1", min_fraud_recall_for_threshold=None)
    run_experiment(strategy_name="class_weight", config=config, splits=splits)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())