"""Central configuration for the BERT fraudulent-job detection module."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# model_code/bert/ → project root = parents[2]
BERT_CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BERT_CODE_ROOT.parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

WEIGHTS_DIR = PROJECT_ROOT / "model_weights" / "bert"
RESULTS_DIR = PROJECT_ROOT / "model_results" / "bert"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
BERT_FINETUNED_DIR = WEIGHTS_DIR

PRETRAINED_MODEL_NAME = "bert-base-uncased"

LABEL_COLUMN = "label"
TEXT_COLUMN = "combined_text"
ID_COLUMN = "record_id"
GROUP_COLUMN = "group_id"
LEGITIMATE_LABEL = 0
FRAUDULENT_LABEL = 1
LABEL_NAMES = {0: "Legitimate", 1: "Fraudulent"}

TEXT_FIELDS: List[str] = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

OPTIONAL_CONTEXT_FIELDS: List[str] = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "location",
    "department",
    "salary_range",
]

TAGGED_FIELD_ORDER: List[str] = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "industry",
    "function",
]

FIELD_TAG_MAP: Dict[str, str] = {
    "title": "TITLE",
    "company_profile": "COMPANY PROFILE",
    "description": "DESCRIPTION",
    "requirements": "REQUIREMENTS",
    "benefits": "BENEFITS",
    "employment_type": "EMPLOYMENT TYPE",
    "required_experience": "REQUIRED EXPERIENCE",
    "industry": "INDUSTRY",
    "function": "FUNCTION",
}

# Prefer tagged section markers; combined_text is retagged when raw fields absent.
PREFER_COMBINED_TEXT = False
USE_TAGGED_FORMAT = True

RANDOM_SEED = 42
# 384 keeps more of long ads on RTX 3060 with batch=4 and accum=4.
MAX_LENGTH = 384
DYNAMIC_PADDING = True

# Softened imbalance defaults (sqrt of inverse-frequency, then clip).
CLASS_WEIGHT_TRANSFORM = "sqrt_clip"
CLASS_WEIGHT_MAX = 5.0
FRAUD_OVERSAMPLE_FACTOR = 3.0
MIN_FRAUD_RECALL_FOR_THRESHOLD = 0.85
THRESHOLD_STEP = 0.01


@dataclass
class BertFinetuneConfig:
    pretrained_model_name: str = PRETRAINED_MODEL_NAME
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 5
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_length: int = MAX_LENGTH
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    dropout: float = 0.1
    random_seed: int = RANDOM_SEED
    fp16: bool = True
    max_grad_norm: float = 1.0
    use_class_weights: bool = True
    class_weight_transform: str = CLASS_WEIGHT_TRANSFORM
    class_weight_max: float = CLASS_WEIGHT_MAX
    fraud_oversample_factor: float = FRAUD_OVERSAMPLE_FACTOR
    gradient_checkpointing: bool = False
    selection_metric: str = "fraud_f1"
    default_threshold: float = 0.5
    optimize_threshold: bool = True
    min_fraud_recall_for_threshold: Optional[float] = MIN_FRAUD_RECALL_FOR_THRESHOLD
    threshold_step: float = THRESHOLD_STEP
    output_dir: str = str(BERT_FINETUNED_DIR)
    run_name: str = "bert_cw_improved"
    model_label: str = "BERT"
    write_canonical_aliases: bool = False


@dataclass
class PredictConfig:
    checkpoint_dir: str = str(BERT_FINETUNED_DIR / "bert_cw_improved" / "best")
    threshold_path: Optional[str] = None
    max_length: int = MAX_LENGTH
    allow_cpu: bool = True


def ensure_directories() -> None:
    for path in (WEIGHTS_DIR, BERT_FINETUNED_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def config_to_dict(cfg: Any) -> Dict[str, Any]:
    if hasattr(cfg, "__dataclass_fields__"):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def default_paths_dict() -> Dict[str, str]:
    return {
        "bert_code_root": str(BERT_CODE_ROOT),
        "project_root": str(PROJECT_ROOT),
        "train_csv": str(TRAIN_CSV),
        "validation_csv": str(VALIDATION_CSV),
        "test_csv": str(TEST_CSV),
        "weights_dir": str(WEIGHTS_DIR),
        "results_dir": str(RESULTS_DIR),
        "figures_dir": str(FIGURES_DIR),
        "logs_dir": str(LOGS_DIR),
        "pretrained_model_name": PRETRAINED_MODEL_NAME,
        "text_fields": TEXT_FIELDS,
        "tagged_field_order": TAGGED_FIELD_ORDER,
        "use_tagged_format": USE_TAGGED_FORMAT,
        "prefer_combined_text": PREFER_COMBINED_TEXT,
    }
