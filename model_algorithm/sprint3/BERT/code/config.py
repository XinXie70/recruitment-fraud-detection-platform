"""Paths, constants, configs, and shared utilities for BERT fraud detection."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# retrain_* /code → experiment root = parent
# Capstone layout: .../capstone-project-.../retrain_*/code
BERT_CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BERT_CODE_ROOT.parent
# paper_aligned_standalone_seed42 lives next to the capstone repo root
REFERENCE_BUNDLE = PROJECT_ROOT.parent.parent / "paper_aligned_standalone_seed42"

# Shared paper-aligned splits live in sibling sprint3/data/
DATA_DIR = PROJECT_ROOT.parent / "data" / "splits"
TRAIN_CSV = DATA_DIR / "train.csv.gz"
VALIDATION_CSV = DATA_DIR / "validation.csv.gz"
TEST_CSV = DATA_DIR / "test.csv.gz"

WEIGHTS_DIR = PROJECT_ROOT / "weight"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
# Kept for import compatibility; process logs are console-only (never under results/).
LOGS_DIR = RESULTS_DIR
BERT_FINETUNED_DIR = WEIGHTS_DIR
REFERENCE_CHECKPOINT = REFERENCE_BUNDLE / "weights" / "best"

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

# Display name written into metrics / comparison tables.
MODEL_LABEL = "Optimized BERT"
# Short slug used in artifact filenames (keep filenames short and stable).
RUN_NAME = "optimized"


def artifact_slug(text: str) -> str:
    """Turn a display label into a filesystem-safe short slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower())
    return slug.strip("_") or "run"


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
    threshold_selection_mode: str = "max_fbeta"
    threshold_f_beta: float = 0.5
    threshold_f1_tolerance: float = 0.01
    min_fraud_recall_for_threshold: Optional[float] = MIN_FRAUD_RECALL_FOR_THRESHOLD
    threshold_step: float = THRESHOLD_STEP
    output_dir: str = str(BERT_FINETUNED_DIR)
    run_name: str = RUN_NAME
    model_label: str = MODEL_LABEL
    write_canonical_aliases: bool = True
    write_error_analysis: bool = True


@dataclass
class PredictConfig:
    checkpoint_dir: str = str(BERT_FINETUNED_DIR / "best")
    threshold_path: Optional[str] = None
    max_length: int = MAX_LENGTH
    allow_cpu: bool = True


def ensure_directories() -> None:
    for path in (WEIGHTS_DIR, BERT_FINETUNED_DIR, RESULTS_DIR, FIGURES_DIR):
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
        "logs_dir": "console_only",
        "pretrained_model_name": PRETRAINED_MODEL_NAME,
        "text_fields": TEXT_FIELDS,
        "tagged_field_order": TAGGED_FIELD_ORDER,
        "use_tagged_format": USE_TAGGED_FORMAT,
        "prefer_combined_text": PREFER_COMBINED_TEXT,
    }


# Ensemble validation export target (sibling under sprint3/)
ENSEMBLE_VAL_COPY = (
    PROJECT_ROOT.parent / "ensemble_BERT_FP" / "results" / "bert_validation_predictions.csv"
)
def set_seed(seed: int = 42) -> None:
    """Fix Python, NumPy, and PyTorch RNG state for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logging(log_name: str = "training.log") -> logging.Logger:
    """Configure console-only logging (no results/logs artifacts)."""
    _ = log_name  # call-site compatibility; no log files are written
    ensure_directories()
    logger = logging.getLogger("bert_fraud")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


def collect_environment_info() -> Dict[str, Any]:
    """Gather Python / PyTorch / CUDA / package versions for run_config."""
    info: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "pytorch_version": None,
        "transformers_version": None,
        "imbalanced_learn_version": None,
        "sklearn_version": None,
        "numpy_version": np.__version__,
        "cuda_available": False,
        "pytorch_cuda_version": None,
        "cudnn_version": None,
        "gpu_name": None,
        "gpu_count": 0,
        "current_training_device": "cpu",
    }
    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["pytorch_cuda_version"] = torch.version.cuda
        info["gpu_count"] = int(torch.cuda.device_count())
        if info["cuda_available"]:
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["current_training_device"] = "cuda"
    except Exception as exc:  # pragma: no cover
        info["pytorch_error"] = str(exc)

    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except Exception as exc:  # pragma: no cover
        info["transformers_error"] = str(exc)

    try:
        import imblearn

        info["imbalanced_learn_version"] = imblearn.__version__
    except Exception as exc:  # pragma: no cover
        info["imbalanced_learn_error"] = str(exc)

    try:
        import sklearn

        info["sklearn_version"] = sklearn.__version__
    except Exception as exc:  # pragma: no cover
        info["sklearn_error"] = str(exc)

    return info


def print_environment_banner(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Print the mandatory environment banner at process start."""
    info = collect_environment_info()
    lines = [
        "======== Environment ========",
        f"Python executable      : {info['python_executable']}",
        f"Python version         : {info['python_version']}",
        f"PyTorch version        : {info['pytorch_version']}",
        f"Transformers version   : {info['transformers_version']}",
        f"CUDA available         : {info['cuda_available']}",
        f"PyTorch CUDA version   : {info['pytorch_cuda_version']}",
        f"cuDNN version          : {info['cudnn_version']}",
        f"GPU name               : {info['gpu_name']}",
        f"GPU count              : {info['gpu_count']}",
        f"Current training device: {info['current_training_device']}",
        "=============================",
    ]
    for line in lines:
        if logger is not None:
            logger.info(line)
        else:
            print(line, flush=True)
    return info


def require_cuda_for_training(logger: Optional[logging.Logger] = None) -> "torch.device":
    """Abort formal training if CUDA is unavailable (do not silently use CPU)."""
    import torch

    info = print_environment_banner(logger)
    if not torch.cuda.is_available():
        msg = (
            "CUDA is not available. Formal training must use the E: CUDA environment.\n"
            f"Current sys.executable = {sys.executable}\n"
            f"PyTorch version = {info.get('pytorch_version')}\n"
            f"CUDA available = False\n"
            "Please activate the E: CUDA env, for example:\n"
            "  . E:\\ml\\activate.ps1\n"
            "  # or run with:\n"
            "  E:\\path\\to\\cuda_environment\\python.exe BERT/code/bert.py train\n"
            "Training aborted (will not fall back to CPU for long runs)."
        )
        if logger is not None:
            logger.error(msg)
        else:
            print(msg, flush=True)
        raise SystemExit(1)
    return torch.device("cuda")


def get_device(allow_cpu: bool = False, logger: Optional[logging.Logger] = None):
    """Return cuda device, or cpu only when explicitly allowed (e.g. predict)."""
    import torch

    print_environment_banner(logger)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        if logger:
            logger.warning("CUDA unavailable; continuing on CPU (allowed for this script).")
        return torch.device("cpu")
    return require_cuda_for_training(logger)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def gpu_memory_mb() -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    except Exception:
        return None


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
