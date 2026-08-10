"""BERT fraudulent job-ad detection — single-file entrypoint.

CLI (from sprint3/):
  python BERT/code/bert.py train
  python BERT/code/bert.py evaluate
  python BERT/code/bert.py predict --text "..."
  python BERT/code/bert.py export-val
"""

from __future__ import annotations



# ========================================================================
# CONFIG
# ========================================================================

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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
LOGS_DIR = RESULTS_DIR / "logs"
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
    run_name: str = "bert_cw_improved"
    model_label: str = "BERT"
    write_canonical_aliases: bool = False


@dataclass
class PredictConfig:
    checkpoint_dir: str = str(BERT_FINETUNED_DIR / "best")
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


# Ensemble validation export target (sibling under sprint3/)
ENSEMBLE_VAL_COPY = (
    PROJECT_ROOT.parent / "ensemble_BERT_FP" / "results" / "bert_validation_predictions.csv"
)


# ========================================================================
# UTILS
# ========================================================================

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np


if TYPE_CHECKING:
    import torch


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
    """Configure console + file logging under models/bert/logs/."""
    ensure_directories()
    logger = logging.getLogger("bert_fraud")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOGS_DIR / log_name, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

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


# ========================================================================
# PREPROCESSING
# ========================================================================

import html
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd



_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n\s*\n+")
_URL = re.compile(
    r"(https?://\S+|www\.\S+)",
    re.IGNORECASE,
)
_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}"
)


def missing_to_empty(value: Any) -> str:
    """Convert missing / literal 'nan' values to empty string."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def clean_text(
    value: Any,
    *,
    replace_url: bool = True,
    replace_email: bool = True,
    replace_phone: bool = False,
) -> str:
    """Light cleaning suitable for BERT (no stemming / stopword removal)."""
    text = missing_to_empty(value)
    if not text:
        return ""

    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub(" ", text)

    if replace_url:
        text = _URL.sub(" [URL] ", text)
    if replace_email:
        text = _EMAIL.sub(" [EMAIL] ", text)
    if replace_phone:
        text = _PHONE.sub(" [PHONE] ", text)

    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n", text)
    return text.strip()


def build_tagged_text(row: Dict[str, Any], field_order: Optional[Iterable[str]] = None) -> str:
    """Concatenate available fields with fixed tags, never dropping the row."""
    order = list(field_order) if field_order is not None else list(TAGGED_FIELD_ORDER)
    parts: List[str] = []
    for field in order:
        if field not in row:
            continue
        cleaned = clean_text(row.get(field, ""))
        if not cleaned:
            continue
        tag = FIELD_TAG_MAP.get(field, field.upper())
        parts.append(f"[{tag}] {cleaned}")
    return "\n".join(parts)


def build_tagged_from_combined(combined: str) -> str:
    """Best-effort section tags when only `combined_text` is available.

    Pipeline text is newline-joined from TEXT_FIELDS with empty fields dropped.
    Exact field recovery is impossible when fields contain newlines, so we map:
      - line 0 -> TITLE
      - line 1 -> COMPANY PROFILE (if present)
      - remaining lines split across DESCRIPTION / REQUIREMENTS / BENEFITS
    """
    text = clean_text(combined, replace_url=True, replace_email=True)
    if not text:
        return ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return f"[TITLE] {lines[0]}"

    parts: List[str] = [f"[TITLE] {lines[0]}"]
    rest = lines[1:]
    if len(rest) == 1:
        parts.append(f"[DESCRIPTION] {rest[0]}")
        return "\n".join(parts)

    parts.append(f"[COMPANY PROFILE] {rest[0]}")
    body = rest[1:]
    if not body:
        return "\n".join(parts)
    if len(body) == 1:
        parts.append(f"[DESCRIPTION] {body[0]}")
        return "\n".join(parts)

    # Split remaining lines into up to three body sections.
    n = len(body)
    cuts = [max(1, n // 3), max(2, (2 * n) // 3)]
    desc = "\n".join(body[: cuts[0]])
    reqs = "\n".join(body[cuts[0] : cuts[1]])
    bens = "\n".join(body[cuts[1] :])
    if desc:
        parts.append(f"[DESCRIPTION] {desc}")
    if reqs:
        parts.append(f"[REQUIREMENTS] {reqs}")
    if bens:
        parts.append(f"[BENEFITS] {bens}")
    return "\n".join(parts)


def build_plain_combined(row: Dict[str, Any], fields: Optional[Iterable[str]] = None) -> str:
    """Newline-join cleaned non-empty fields (matches shared data contract)."""
    order = list(fields) if fields is not None else list(TEXT_FIELDS)
    parts = [clean_text(row.get(field, "")) for field in order]
    return "\n".join(p for p in parts if p)


def extract_text_from_row(
    row: Dict[str, Any],
    *,
    prefer_combined: bool = PREFER_COMBINED_TEXT,
    use_tagged: bool = USE_TAGGED_FORMAT,
) -> str:
    """Resolve model input text for one advertisement row."""
    available = [f for f in TAGGED_FIELD_ORDER if f in row and missing_to_empty(row.get(f, ""))]

    # Tagged format takes priority when enabled.
    if use_tagged:
        if available:
            return build_tagged_text(row)
        if TEXT_COLUMN in row:
            tagged = build_tagged_from_combined(row.get(TEXT_COLUMN, ""))
            if tagged:
                return tagged

    if prefer_combined and TEXT_COLUMN in row:
        combined = clean_text(row.get(TEXT_COLUMN, ""), replace_url=True, replace_email=True)
        if combined:
            return combined

    if available:
        return build_plain_combined(row, available)

    if TEXT_COLUMN in row:
        combined = clean_text(row.get(TEXT_COLUMN, ""), replace_url=True, replace_email=True)
        if combined:
            return combined

    # Fallback: any string-like leftover columns except ids/labels.
    skip = {
        "label",
        "fraudulent",
        "record_id",
        "group_id",
        "in_balanced_dataset",
        "model_text",
        "split",
        "row_index",
    }
    leftovers = [
        clean_text(v)
        for k, v in row.items()
        if k not in skip and isinstance(v, (str, int, float))
    ]
    return "\n".join(x for x in leftovers if x)


def prepare_dataframe_text(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a standardised `model_text` column."""
    out = df.copy()
    texts = [
        extract_text_from_row(row._asdict() if hasattr(row, "_asdict") else row.to_dict())
        for _, row in out.iterrows()
    ]
    out["model_text"] = texts
    return out


def derive_title_preview(text: str, max_chars: int = 120) -> str:
    """Best-effort title from first line of combined text."""
    if not text:
        return ""
    first = text.split("\n", 1)[0].strip()
    return first[:max_chars]


def derive_company_preview(text: str, max_chars: int = 120) -> str:
    """Best-effort company snippet from second non-empty line."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return ""
    return lines[1][:max_chars]


# ========================================================================
# METRICS
# ========================================================================

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute a full metric suite focused on the fraud class (label=1)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    metrics: Dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "legitimate_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "legitimate_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "legitimate_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "fraud_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraudulent"],
            zero_division=0,
            output_dict=True,
        ),
    }

    # ROC / PR need both classes present in y_true for a meaningful score.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def threshold_sweep(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
    *,
    step: float = 0.01,
) -> pd.DataFrame:
    """Evaluate fraud metrics across candidate thresholds (default step 0.01)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, float(step)), 4)
        # Always include the endpoints used historically for comparability.
        thresholds = sorted({0.01, *thresholds.tolist(), 0.99})
    rows = []
    for thr in thresholds:
        m = binary_metrics(y_true, y_prob, threshold=float(thr))
        rows.append(
            {
                "threshold": float(thr),
                "fraud_precision": m["fraud_precision"],
                "fraud_recall": m["fraud_recall"],
                "fraud_f1": m["fraud_f1"],
                "false_positive": m["confusion_matrix"]["fp"],
                "false_negative": m["confusion_matrix"]["fn"],
                "tn": m["confusion_matrix"]["tn"],
                "tp": m["confusion_matrix"]["tp"],
                "macro_f1": m["macro_f1"],
                "accuracy": m["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    mode: str = "max_fraud_f1",
    min_fraud_recall: Optional[float] = None,
    step: float = 0.01,
    f_beta: float = 0.5,
    f1_tolerance: float = 0.01,
) -> Tuple[float, pd.DataFrame]:
    """Choose a classification threshold on the VALIDATION set only.

    Modes:
      - max_fraud_f1: threshold with highest fraud F1
        (ties broken by higher precision, then higher threshold)
      - max_fbeta: maximize fraud F-beta (beta<1 favors precision / higher thr)
      - near_max_f1_prefer_precision: among thresholds within f1_tolerance of
        the best fraud F1, pick highest fraud precision then highest threshold
      - min_recall_then_f1: among thresholds with fraud_recall >= min_fraud_recall,
        pick highest fraud F1 (falls back to unconstrained max F1 if none qualify)
      - min_recall_then_precision: among thresholds meeting min fraud recall,
        pick highest fraud precision (falls back to max fraud F1 if none qualify)
    """
    sweep = threshold_sweep(y_true, y_prob, step=step)

    if mode == "min_recall_then_f1" and min_fraud_recall is not None:
        eligible = sweep[sweep["fraud_recall"] >= float(min_fraud_recall)]
        if len(eligible) > 0:
            best = eligible.sort_values(
                ["fraud_f1", "fraud_precision", "threshold"],
                ascending=[False, False, False],
            ).iloc[0]
            return float(best["threshold"]), sweep

    if mode == "min_recall_then_precision" and min_fraud_recall is not None:
        eligible = sweep[sweep["fraud_recall"] >= float(min_fraud_recall)]
        if len(eligible) > 0:
            best = eligible.sort_values(
                ["fraud_precision", "fraud_f1", "threshold"],
                ascending=[False, False, False],
            ).iloc[0]
            return float(best["threshold"]), sweep

    if mode == "max_fbeta":
        beta = float(f_beta)
        beta2 = beta * beta
        scored = sweep.copy()
        denom = beta2 * scored["fraud_precision"] + scored["fraud_recall"]
        scored["fraud_fbeta"] = np.where(
            denom > 0,
            (1.0 + beta2) * scored["fraud_precision"] * scored["fraud_recall"] / denom,
            0.0,
        )
        best = scored.sort_values(
            ["fraud_fbeta", "fraud_precision", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        return float(best["threshold"]), sweep

    if mode == "near_max_f1_prefer_precision":
        max_f1 = float(sweep["fraud_f1"].max())
        eligible = sweep[sweep["fraud_f1"] >= max_f1 - float(f1_tolerance)]
        best = eligible.sort_values(
            ["fraud_precision", "threshold", "fraud_f1"],
            ascending=[False, False, False],
        ).iloc[0]
        return float(best["threshold"]), sweep

    # Default: max fraud F1; prefer precision / higher threshold on ties.
    best = sweep.sort_values(
        ["fraud_f1", "fraud_precision", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best["threshold"]), sweep


def build_predictions_frame(
    *,
    record_ids: Sequence[str],
    titles: Sequence[str],
    companies: Sequence[str],
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float,
    model_name: str,
    imbalance_strategy: str,
    row_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    if row_indices is None:
        row_indices = list(range(n))
    return pd.DataFrame(
        {
            "sample_index": list(row_indices),
            "original_id": list(record_ids),
            "company": list(companies),
            "title": list(titles),
            "true_label": y_true,
            "predicted_label": y_pred,
            "legitimate_probability": 1.0 - y_prob,
            "fraud_probability": y_prob,
            "threshold": threshold,
            "correct": (y_true == y_pred).astype(int),
            "model_name": model_name,
            "imbalance_strategy": imbalance_strategy,
        }
    )


def build_error_analysis(
    preds: pd.DataFrame,
    texts: Sequence[str],
    *,
    high_conf: float = 0.85,
    near_margin: float = 0.05,
) -> pd.DataFrame:
    """Flag FP / FN / high-confidence errors / near-threshold samples."""
    df = preds.copy()
    df["text_preview"] = [str(t)[:300].replace("\n", " ") for t in texts]
    thr = float(df["threshold"].iloc[0]) if len(df) else 0.5

    def error_type(row) -> str:
        if row["true_label"] == 0 and row["predicted_label"] == 1:
            base = "false_positive"
        elif row["true_label"] == 1 and row["predicted_label"] == 0:
            base = "false_negative"
        else:
            base = "correct"
        extras = []
        if base != "correct" and (
            (row["predicted_label"] == 1 and row["fraud_probability"] >= high_conf)
            or (row["predicted_label"] == 0 and row["fraud_probability"] <= 1 - high_conf)
        ):
            extras.append("high_confidence_error")
        if abs(row["fraud_probability"] - thr) <= near_margin:
            extras.append("near_threshold")
        if extras:
            return base + "|" + "|".join(extras)
        return base

    df["error_type"] = df.apply(error_type, axis=1)
    df = df.rename(
        columns={
            "original_id": "sample_id",
            "true_label": "original_label",
        }
    )
    cols = [
        "sample_id",
        "title",
        "company",
        "original_label",
        "predicted_label",
        "fraud_probability",
        "threshold",
        "error_type",
        "text_preview",
    ]
    return df[cols]


def save_comparison_row(
    path: Path,
    row: Dict[str, Any],
) -> pd.DataFrame:
    """Append / upsert one model row into model_comparison.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "Model",
        "Imbalance method",
        "Accuracy",
        "Fraud Precision",
        "Fraud Recall",
        "Fraud F1",
        "Macro F1",
        "ROC-AUC",
        "PR-AUC",
        "Inference time (s)",
        "Notes",
    ]
    if path.exists():
        table = pd.read_csv(path)
    else:
        table = pd.DataFrame(columns=cols)

    key_model = row.get("Model")
    key_imb = row.get("Imbalance method")
    mask = (table["Model"] == key_model) & (table["Imbalance method"] == key_imb)
    for c in cols:
        if c not in table.columns:
            table[c] = np.nan
    if mask.any():
        for k, v in row.items():
            table.loc[mask, k] = v
    else:
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)
    return table


def plot_training_curves(history: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "train_loss" in history.columns:
        ax.plot(history["epoch"], history["train_loss"], label="train_loss")
    if "val_loss" in history.columns:
        ax.plot(history["epoch"], history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: Dict[str, Any], out_path: Path, title: str = "Confusion Matrix") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = np.asarray(cm["matrix"] if isinstance(cm, dict) and "matrix" in cm else cm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Legit", "Pred Fraud"],
        yticklabels=["True Legit", "True Fraud"],
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_pr(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    roc_path: Path,
    pr_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    roc_path.parent.mkdir(parents=True, exist_ok=True)

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(y_true, y_prob):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, label=f"PR-AUC={average_precision_score(y_true, y_prob):.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)


def plot_class_distribution(splits: Dict[str, pd.DataFrame], out_path: Path, label_col: str = "label") -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    names, legit, fraud = [], [], []
    for name, df in splits.items():
        names.append(name)
        legit.append(int((df[label_col] == 0).sum()))
        fraud.append(int((df[label_col] == 1).sum()))
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, legit, width, label="Legitimate")
    ax.bar(x + width / 2, fraud, width, label="Fraudulent")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution by Split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(comparison_csv: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not comparison_csv.exists():
        return
    df = pd.read_csv(comparison_csv)
    if df.empty:
        return
    # Drop unavailable rows from bar chart values but keep labels.
    plot_df = df.copy()
    for col in ["Fraud Recall", "Fraud F1", "PR-AUC"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    labels = [
        f"{m}\n({i})"
        for m, i in zip(plot_df["Model"], plot_df["Imbalance method"], strict=True)
    ]
    x = np.arange(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.2), 5))
    ax.bar(x - width, plot_df["Fraud Recall"], width, label="Fraud Recall")
    ax.bar(x, plot_df["Fraud F1"], width, label="Fraud F1")
    ax.bar(x + width, plot_df["PR-AUC"], width, label="PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison (fraud-focused)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ========================================================================
# MODEL
# ========================================================================

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification


class BertForFraudClassification(nn.Module):
    """Thin wrapper around AutoModelForSequenceClassification."""

    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int = 2,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dropout
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            config.classifier_dropout = dropout
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=config,
        )
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return self.model(**kwargs)

    @property
    def encoder(self):
        return self.model.bert if hasattr(self.model, "bert") else self.model.base_model


def softmax_fraud_proba(logits):
    """Return P(fraud=1) from 2-class logits."""
    return F.softmax(logits, dim=-1)[:, 1]


# ========================================================================
# DATASET
# ========================================================================

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset



SPLIT_FILES = {
    "train": TRAIN_CSV,
    "validation": VALIDATION_CSV,
    "test": TEST_CSV,
}


def load_split(split: str, path: Optional[Path] = None) -> pd.DataFrame:
    """Load one fixed split CSV. Does not merge or re-partition."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_FILES)}")
    csv_path = Path(path) if path is not None else SPLIT_FILES[split]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Required split file missing: {csv_path}. "
            "Do not create a new split; use the project train/validation/test CSVs."
        )
    df = pd.read_csv(csv_path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{csv_path} missing required column '{LABEL_COLUMN}'")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    # Paper-aligned splits already store the exact model_text used in training.
    # Do not rebuild it — retagging would change the input and break reproduction.
    if "model_text" not in df.columns or df["model_text"].isna().all():
        df = prepare_dataframe_text(df)
    else:
        df["model_text"] = df["model_text"].fillna("").astype(str)
    df["split"] = split
    df["row_index"] = np.arange(len(df), dtype=np.int64)
    if ID_COLUMN not in df.columns:
        df[ID_COLUMN] = [f"{split}_{i:05d}" for i in range(len(df))]
    if "title" not in df.columns:
        df["title"] = df["model_text"].map(derive_title_preview)
    if "company" not in df.columns:
        # Splits do not store company separately; derive a preview for analysis.
        df["company"] = df["model_text"].map(derive_company_preview)
    return df


def load_all_splits(
    train_path: Optional[Path] = None,
    validation_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Load the three fixed splits independently (no merging)."""
    return {
        "train": load_split("train", train_path),
        "validation": load_split("validation", validation_path),
        "test": load_split("test", test_path),
    }


def _label_stats(df: pd.DataFrame) -> Dict[str, Any]:
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    n0 = int(counts.get(0, 0))
    n1 = int(counts.get(1, 0))
    total = len(df)
    return {
        "n_samples": total,
        "n_legitimate": n0,
        "n_fraudulent": n1,
        "class_ratio_legitimate": float(n0 / total) if total else 0.0,
        "class_ratio_fraudulent": float(n1 / total) if total else 0.0,
        "label_counts": {str(k): int(v) for k, v in sorted(counts.items())},
    }


def _text_stats(df: pd.DataFrame) -> Dict[str, Any]:
    texts = df["model_text"].fillna("")
    lengths = texts.str.len()
    return {
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "empty_text_count": int((texts.str.strip() == "").sum()),
        "exact_duplicate_text_count": int(texts.duplicated().sum()),
        "duplicate_id_count": int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df.columns else None,
        "text_length": {
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "median": float(lengths.median()) if len(lengths) else 0.0,
            "min": int(lengths.min()) if len(lengths) else 0,
            "max": int(lengths.max()) if len(lengths) else 0,
            "p90": float(lengths.quantile(0.90)) if len(lengths) else 0.0,
            "p95": float(lengths.quantile(0.95)) if len(lengths) else 0.0,
            "p99": float(lengths.quantile(0.99)) if len(lengths) else 0.0,
        },
        "columns": list(df.columns),
        "title_unique": int(df["title"].nunique()) if "title" in df.columns else None,
        "company_unique": int(df["company"].nunique()) if "company" in df.columns else None,
    }


def compute_token_length_stats(
    texts: Sequence[str],
    tokenizer,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Tokenise without truncation to estimate length distribution."""
    sample = list(texts)
    if max_samples is not None and len(sample) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(sample), size=max_samples, replace=False)
        sample = [sample[i] for i in idx]

    lengths: List[int] = []
    for text in sample:
        enc = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        lengths.append(len(enc["input_ids"]))

    arr = np.asarray(lengths, dtype=np.float64)

    def pct_over(limit: int) -> float:
        return float((arr > limit).mean()) if len(arr) else 0.0

    p_over_256 = pct_over(256)
    p_over_384 = pct_over(384)
    p_over_512 = pct_over(512)
    reason = (
        f"Measured train token lengths: median≈{float(np.median(arr)) if len(arr) else 0:.0f}, "
        f"{p_over_256:.1%} >256, {p_over_384:.1%} >384, {p_over_512:.1%} >512. "
        "Default max_length=384 is chosen for RTX 3060 with batch=4 / accum=4; "
        "try 512 only after an OOM check. Long ads may still be truncated."
    )
    return {
        "n_texts_measured": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
        "pct_over_128": pct_over(128),
        "pct_over_256": p_over_256,
        "pct_over_384": p_over_384,
        "pct_over_512": p_over_512,
        "recommended_max_length": 384,
        "recommendation_reason": reason,
    }


def build_data_quality_report(
    splits: Dict[str, pd.DataFrame],
    token_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce a report only — never re-split based on findings."""
    per_split = {}
    for name, df in splits.items():
        per_split[name] = {**_label_stats(df), **_text_stats(df)}

    train, val, test = splits["train"], splits["validation"], splits["test"]

    def set_overlap(a: pd.Series, b: pd.Series) -> int:
        return int(len(set(a.astype(str)) & set(b.astype(str))))

    cross = {
        "identical_text_train_validation": set_overlap(train["model_text"], val["model_text"]),
        "identical_text_train_test": set_overlap(train["model_text"], test["model_text"]),
        "identical_text_validation_test": set_overlap(val["model_text"], test["model_text"]),
        "identical_id_train_validation": set_overlap(train[ID_COLUMN], val[ID_COLUMN]),
        "identical_id_train_test": set_overlap(train[ID_COLUMN], test[ID_COLUMN]),
        "identical_id_validation_test": set_overlap(val[ID_COLUMN], test[ID_COLUMN]),
        "note": (
            "Cross-split overlaps are reported for awareness only. "
            "This module never reassigns rows across splits."
        ),
    }

    if GROUP_COLUMN in train.columns:
        cross.update(
            {
                "group_overlap_train_validation": set_overlap(train[GROUP_COLUMN], val[GROUP_COLUMN]),
                "group_overlap_train_test": set_overlap(train[GROUP_COLUMN], test[GROUP_COLUMN]),
                "group_overlap_validation_test": set_overlap(val[GROUP_COLUMN], test[GROUP_COLUMN]),
            }
        )

    # Company+title combo check when both derived/present.
    if "company" in train.columns and "title" in train.columns:
        def combo(df: pd.DataFrame) -> pd.Series:
            return df["company"].astype(str) + " || " + df["title"].astype(str)

        cross["company_title_overlap_train_validation"] = set_overlap(combo(train), combo(val))
        cross["company_title_overlap_train_test"] = set_overlap(combo(train), combo(test))
        cross["company_title_overlap_validation_test"] = set_overlap(combo(val), combo(test))

    report: Dict[str, Any] = {
        "split_files": {k: str(SPLIT_FILES[k]) for k in SPLIT_FILES},
        "per_split": per_split,
        "cross_split_checks": cross,
        "policy": {
            "no_resplit": True,
            "no_merge": True,
            "no_cross_validation": True,
            "oversample_train_only": True,
        },
    }
    if token_stats is not None:
        report["token_length_stats_train"] = token_stats
    return report


def compute_class_weights(
    y: np.ndarray,
    n_classes: int = 2,
    *,
    transform: str = "none",
    weight_max: Optional[float] = None,
) -> np.ndarray:
    """Inverse-frequency weights from TRAIN labels only.

    weight[c] = N / (n_classes * count[c])

    Optional softening:
      - sqrt: square-root of inverse-frequency weights
      - sqrt_clip: sqrt then clip to [1/weight_max, weight_max] (or [0, weight_max] if max set)
      - clip: clip raw inverse-frequency to weight_max
      - none: raw inverse-frequency
    """
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    total = float(len(y))
    weights = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        if counts[c] > 0:
            weights[c] = total / (n_classes * counts[c])
        else:
            weights[c] = 1.0

    transform = (transform or "none").strip().lower()
    if transform in {"sqrt", "sqrt_clip"}:
        weights = np.sqrt(weights)
    if transform in {"clip", "sqrt_clip"} and weight_max is not None and weight_max > 0:
        weights = np.clip(weights, 1.0 / float(weight_max), float(weight_max))
    return weights


def oversample_fraud_train(
    df: pd.DataFrame,
    *,
    factor: float = 3.0,
    seed: int = 42,
    label_column: str = LABEL_COLUMN,
    fraud_label: int = 1,
) -> pd.DataFrame:
    """Train-only minority oversampling. Validation/test must never be passed here.

    ``factor`` multiplies the fraud row count (e.g. 3.0 => ~3x fraud examples).
    """
    if factor is None or float(factor) <= 1.0:
        return df.reset_index(drop=True)

    factor = float(factor)
    fraud = df[df[label_column] == fraud_label]
    if fraud.empty:
        return df.reset_index(drop=True)

    n_target = int(round(len(fraud) * factor))
    extra = n_target - len(fraud)
    if extra <= 0:
        return df.reset_index(drop=True)

    rng = np.random.RandomState(int(seed))
    sampled = fraud.sample(n=extra, replace=True, random_state=rng)
    out = pd.concat([df, sampled], ignore_index=True)
    return out.sample(frac=1.0, random_state=rng).reset_index(drop=True)


class JobTextDataset(Dataset):
    """Tokenised job-ad dataset for end-to-end BERT fine-tuning."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 256,
        record_ids: Optional[Sequence[str]] = None,
        titles: Optional[Sequence[str]] = None,
        companies: Optional[Sequence[str]] = None,
    ) -> None:
        self.texts = list(texts)
        self.labels = [int(x) for x in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.record_ids = list(record_ids) if record_ids is not None else [str(i) for i in range(len(self.texts))]
        self.titles = list(titles) if titles is not None else [""] * len(self.texts)
        self.companies = list(companies) if companies is not None else [""] * len(self.texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.labels[idx],
            "record_id": self.record_ids[idx],
            "title": self.titles[idx],
            "company": self.companies[idx],
            "row_index": idx,
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"]
        return item


def dataframe_to_text_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> JobTextDataset:
    return JobTextDataset(
        texts=df["model_text"].tolist(),
        labels=df[LABEL_COLUMN].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        record_ids=df[ID_COLUMN].astype(str).tolist(),
        titles=df["title"].astype(str).tolist() if "title" in df.columns else None,
        companies=df["company"].astype(str).tolist() if "company" in df.columns else None,
    )


# ========================================================================
# TRAINING
# ========================================================================

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from tqdm import tqdm



def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def collate_batch(features: List[Dict[str, Any]], collator: DataCollatorWithPadding) -> Dict[str, Any]:
    meta_keys = ("record_id", "title", "company", "row_index")
    meta = {k: [f[k] for f in features] for k in meta_keys}
    model_feats = []
    for f in features:
        item = {
            "input_ids": f["input_ids"],
            "attention_mask": f["attention_mask"],
            "labels": f["labels"],
        }
        if "token_type_ids" in f:
            item["token_type_ids"] = f["token_type_ids"]
        model_feats.append(item)
    batch = collator(model_feats)
    batch.update(meta)
    return batch


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    use_amp: bool = True,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[str], List[str], List[str], List[int]]:
    model.eval()
    losses: List[float] = []
    probs: List[float] = []
    labels: List[int] = []
    record_ids: List[str] = []
    titles: List[str] = []
    companies: List[str] = []
    row_indices: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with autocast(enabled=use_amp and device.type == "cuda"):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            if criterion is not None:
                loss = criterion(logits, y)
                losses.append(float(loss.item()))

        batch_prob = softmax_fraud_proba(logits).detach().float().cpu().numpy()
        probs.extend(batch_prob.tolist())
        labels.extend(y.detach().cpu().numpy().tolist())
        record_ids.extend(batch["record_id"])
        titles.extend(batch["title"])
        companies.extend(batch["company"])
        row_indices.extend([int(x) for x in batch["row_index"]])

    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    metrics = binary_metrics(y_true, y_prob, threshold=0.5)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, y_true, y_prob, record_ids, titles, companies, row_indices


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    criterion: nn.Module,
    scaler: GradScaler,
    grad_accum: int,
    max_grad_norm: float,
    use_amp: bool,
) -> float:
    model.train()
    running = 0.0
    n_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        try:
            with autocast(enabled=use_amp and device.type == "cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                loss = criterion(outputs.logits, y) / grad_accum
            scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during BERT fine-tuning. Try in order:\n"
                "  1) decrease --train_batch_size\n"
                "  2) increase --gradient_accumulation_steps\n"
                "  3) decrease --max_length\n"
                "  4) enable --gradient_checkpointing\n"
                "Do NOT silently continue on CPU for long training."
            ) from exc

        running += float(loss.item()) * grad_accum
        n_steps += 1

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

    return running / max(n_steps, 1)


def save_checkpoint(
    output_dir: Path,
    model: BertForFraudClassification,
    tokenizer,
    cfg: BertFinetuneConfig,
    extra: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_json({"config": config_to_dict(cfg), **extra}, output_dir / "train_meta.json")


def run_training(
    cfg: BertFinetuneConfig,
    splits: Optional[Dict[str, pd.DataFrame]] = None,
    results_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    log_name: str = "training.log",
) -> Dict[str, Any]:
    ensure_directories()
    results_root = Path(results_dir) if results_dir is not None else RESULTS_DIR
    figures_root = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
    results_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_name)
    set_seed(cfg.random_seed)
    device = require_cuda_for_training(logger)

    if splits is None:
        logger.info("Loading fixed splits (no re-split, no CV, no merge).")
        splits = load_all_splits()
    else:
        required = {"train", "validation", "test"}
        missing = required - set(splits)
        if missing:
            raise ValueError(f"splits missing keys: {sorted(missing)}")
        logger.info(
            "Using caller-provided splits: train=%s validation=%s test=%s",
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["test"]),
        )

    plot_class_distribution(splits, figures_root / "class_distribution.png")

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name)
    token_stats = compute_token_length_stats(
        splits["train"]["model_text"].tolist(),
        tokenizer,
        max_samples=min(3000, len(splits["train"])),
    )
    logger.info("Token length stats (train sample): %s", token_stats)
    quality = build_data_quality_report(splits, token_stats=token_stats)
    save_json(quality, results_root / "data_quality_report.json")
    logger.info("Saved data_quality_report.json")

    train_frame = splits["train"]
    n_train_before = len(train_frame)
    n_fraud_before = int((train_frame[LABEL_COLUMN] == 1).sum())
    train_frame = oversample_fraud_train(
        train_frame,
        factor=cfg.fraud_oversample_factor,
        seed=cfg.random_seed,
    )
    n_train_after = len(train_frame)
    n_fraud_after = int((train_frame[LABEL_COLUMN] == 1).sum())
    logger.info(
        "Train-only fraud oversample: factor=%.2f  rows %d→%d  fraud %d→%d",
        float(cfg.fraud_oversample_factor),
        n_train_before,
        n_train_after,
        n_fraud_before,
        n_fraud_after,
    )

    class_weights = compute_class_weights(
        train_frame[LABEL_COLUMN].to_numpy(),
        transform=cfg.class_weight_transform,
        weight_max=cfg.class_weight_max,
    )
    logger.info(
        "Train-only class weights (transform=%s, max=%s): %s",
        cfg.class_weight_transform,
        cfg.class_weight_max,
        class_weights.tolist(),
    )

    train_ds = dataframe_to_text_dataset(train_frame, tokenizer, cfg.max_length)
    val_ds = dataframe_to_text_dataset(splits["validation"], tokenizer, cfg.max_length)
    test_ds = dataframe_to_text_dataset(splits["test"], tokenizer, cfg.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def make_loader(ds, batch_size, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda feats: collate_batch(feats, collator),
            num_workers=0,
        )

    train_loader = make_loader(train_ds, cfg.train_batch_size, True)
    val_loader = make_loader(val_ds, cfg.eval_batch_size, False)
    test_loader = make_loader(test_ds, cfg.eval_batch_size, False)

    model = BertForFraudClassification(
        cfg.pretrained_model_name,
        num_labels=2,
        dropout=cfg.dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(device)

    if cfg.use_class_weights:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        imbalance_name = (
            f"Class weight ({cfg.class_weight_transform})"
            if cfg.fraud_oversample_factor <= 1.0
            else f"Oversample×{cfg.fraud_oversample_factor:g}+CW({cfg.class_weight_transform})"
        )
    else:
        criterion = nn.CrossEntropyLoss()
        imbalance_name = (
            "None"
            if cfg.fraud_oversample_factor <= 1.0
            else f"Oversample×{cfg.fraud_oversample_factor:g}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_update_steps = (
        (len(train_loader) + cfg.gradient_accumulation_steps - 1)
        // cfg.gradient_accumulation_steps
    ) * cfg.num_train_epochs
    warmup_steps = int(total_update_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    scaler = GradScaler(enabled=cfg.fp16 and device.type == "cuda")

    history_rows: List[Dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = -1
    patience_left = cfg.early_stopping_patience
    # Checkpoints are stored directly under output_dir/best (no run_name nesting).
    output_dir = Path(cfg.output_dir)
    best_dir = output_dir / "best"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_timer = Timer()
    logger.info(
        "Start fine-tuning: model=%s use_class_weights=%s max_length=%s batch=%s accum=%s",
        cfg.pretrained_model_name,
        cfg.use_class_weights,
        cfg.max_length,
        cfg.train_batch_size,
        cfg.gradient_accumulation_steps,
    )

    for epoch in range(1, cfg.num_train_epochs + 1):
        epoch_timer = Timer()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            criterion,
            scaler,
            cfg.gradient_accumulation_steps,
            cfg.max_grad_norm,
            cfg.fp16,
        )
        val_metrics, _, _, _, _, _, _ = evaluate_loader(
            model, val_loader, device, criterion=criterion, use_amp=cfg.fp16
        )
        lr_now = float(scheduler.get_last_lr()[0])
        mem = gpu_memory_mb()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "legitimate_precision": val_metrics["legitimate_precision"],
            "legitimate_recall": val_metrics["legitimate_recall"],
            "legitimate_f1": val_metrics["legitimate_f1"],
            "fraud_precision": val_metrics["fraud_precision"],
            "fraud_recall": val_metrics["fraud_recall"],
            "fraud_f1": val_metrics["fraud_f1"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "roc_auc": val_metrics["roc_auc"],
            "pr_auc": val_metrics["pr_auc"],
            "learning_rate": lr_now,
            "gpu_memory_mb": mem,
            "epoch_duration_sec": epoch_timer.elapsed(),
            "best_val_fraud_f1": max(best_metric, val_metrics["fraud_f1"]),
        }
        history_rows.append(row)
        logger.info(
            "Epoch %s | train_loss=%.4f val_loss=%.4f fraud_f1=%.4f fraud_recall=%.4f pr_auc=%.4f",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["fraud_f1"],
            val_metrics["fraud_recall"],
            val_metrics["pr_auc"],
        )

        selection_value = float(val_metrics.get(cfg.selection_metric, val_metrics["fraud_f1"]))
        if selection_value > best_metric:
            best_metric = selection_value
            best_epoch = epoch
            patience_left = cfg.early_stopping_patience
            save_checkpoint(
                best_dir,
                model,
                tokenizer,
                cfg,
                {
                    "best_epoch": best_epoch,
                    "best_validation_metric": best_metric,
                    "selection_metric": cfg.selection_metric,
                    "imbalance_method": imbalance_name,
                    "class_weights": class_weights.tolist(),
                },
            )
            logger.info("New best checkpoint saved at epoch %s (%s=%.4f)", epoch, cfg.selection_metric, best_metric)
        else:
            patience_left -= 1
            logger.info("No improvement. Patience left: %s", patience_left)
            if patience_left <= 0:
                logger.info("Early stopping at epoch %s", epoch)
                break

    history_df = pd.DataFrame(history_rows)
    history_path = results_root / f"training_history_{cfg.run_name}.csv"
    history_df.to_csv(history_path, index=False)
    write_aliases = cfg.write_canonical_aliases and cfg.use_class_weights
    # Also keep a canonical name for the default/main BERT run.
    if write_aliases:
        history_df.to_csv(results_root / "training_history.csv", index=False)
    plot_training_curves(history_df, figures_root / f"training_loss_{cfg.run_name}.png")
    if write_aliases:
        plot_training_curves(history_df, figures_root / "training_loss.png")

    # Reload best checkpoint before thresholding / test.
    logger.info("Loading best checkpoint from %s", best_dir)
    model.model = type(model.model).from_pretrained(best_dir)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(best_dir)

    # Validation threshold selection (never use test).
    val_metrics_05, y_val, p_val, val_ids, val_titles, val_companies, val_idx = evaluate_loader(
        model, val_loader, device, criterion=criterion, use_amp=cfg.fp16
    )
    mode = "fixed"
    if cfg.optimize_threshold:
        mode = getattr(cfg, "threshold_selection_mode", None) or "max_fraud_f1"
        if cfg.min_fraud_recall_for_threshold is not None and mode in {
            "max_fraud_f1",
            "max_fbeta",
            "near_max_f1_prefer_precision",
        }:
            # Preserve legacy behavior when a recall floor is explicitly set
            # and the caller did not already request a recall-constrained mode.
            if mode == "max_fraud_f1":
                mode = "min_recall_then_f1"
        best_thr, sweep = select_threshold(
            y_val,
            p_val,
            mode=mode,
            min_fraud_recall=cfg.min_fraud_recall_for_threshold,
            step=cfg.threshold_step,
            f_beta=getattr(cfg, "threshold_f_beta", 0.5),
            f1_tolerance=getattr(cfg, "threshold_f1_tolerance", 0.01),
        )
    else:
        best_thr = cfg.default_threshold
        sweep = None
    if sweep is not None:
        sweep.to_csv(results_root / f"threshold_sweep_{cfg.run_name}.csv", index=False)

    val_metrics_opt = binary_metrics(y_val, p_val, threshold=best_thr)
    logger.info(
        "Validation threshold selected on validation only: %.4f "
        "(fraud_f1=%.4f precision=%.4f recall=%.4f mode=%s f_beta=%s min_recall=%s step=%s)",
        best_thr,
        val_metrics_opt["fraud_f1"],
        val_metrics_opt["fraud_precision"],
        val_metrics_opt["fraud_recall"],
        mode,
        getattr(cfg, "threshold_f_beta", None),
        cfg.min_fraud_recall_for_threshold,
        cfg.threshold_step,
    )

    # Final test evaluation once.
    infer_timer = Timer()
    test_metrics_05, y_test, p_test, test_ids, test_titles, test_companies, test_idx = evaluate_loader(
        model, test_loader, device, criterion=criterion, use_amp=cfg.fp16
    )
    inference_time = infer_timer.elapsed()
    test_metrics = binary_metrics(y_test, p_test, threshold=best_thr)
    test_metrics["loss"] = test_metrics_05["loss"]
    test_metrics["threshold_default_0_5"] = binary_metrics(y_test, p_test, threshold=0.5)
    test_metrics["threshold_optimized"] = {
        "threshold": best_thr,
        **{k: v for k, v in test_metrics.items() if k not in {"threshold_default_0_5", "threshold_optimized"}},
    }

    preds = build_predictions_frame(
        record_ids=test_ids,
        titles=test_titles,
        companies=test_companies,
        y_true=y_test,
        y_prob=p_test,
        threshold=best_thr,
        model_name=f"{cfg.model_label}:{cfg.run_name}",
        imbalance_strategy=imbalance_name,
        row_indices=test_idx,
    )
    pred_path = results_root / f"predictions_{cfg.run_name}.csv"
    preds.to_csv(pred_path, index=False)
    if write_aliases:
        preds.to_csv(results_root / "predictions.csv", index=False)

    errors = build_error_analysis(preds, splits["test"]["model_text"].tolist())
    err_path = results_root / f"error_analysis_{cfg.run_name}.csv"
    errors.to_csv(err_path, index=False)
    if write_aliases:
        errors.to_csv(results_root / "error_analysis.csv", index=False)

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        figures_root / f"confusion_matrix_{cfg.run_name}.png",
        title=f"{cfg.model_label} test CM ({cfg.run_name})",
    )
    if write_aliases:
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            figures_root / "confusion_matrix.png",
            title="BERT test Confusion Matrix",
        )
    plot_roc_pr(
        y_test,
        p_test,
        figures_root / f"roc_curve_{cfg.run_name}.png",
        figures_root / f"precision_recall_curve_{cfg.run_name}.png",
    )
    if write_aliases:
        plot_roc_pr(
            y_test,
            p_test,
            figures_root / "roc_curve.png",
            figures_root / "precision_recall_curve.png",
        )

    metrics_stem = "bert" if cfg.model_label.upper() == "BERT" else cfg.model_label.lower()
    metrics_path = results_root / f"{metrics_stem}_test_metrics_{cfg.run_name}.json"
    payload = {
        "model": f"{cfg.model_label} fine-tuned",
        "run_name": cfg.run_name,
        "imbalance_method": imbalance_name,
        "pretrained_model_name": cfg.pretrained_model_name,
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric,
        "selection_metric": cfg.selection_metric,
        "threshold": best_thr,
        "validation_metrics_threshold_0_5": val_metrics_05,
        "validation_metrics_optimized_threshold": val_metrics_opt,
        "test_metrics": test_metrics,
        "inference_time_sec": inference_time,
        "training_time_sec": total_timer.elapsed(),
        "class_weights_train_only": class_weights.tolist(),
    }
    save_json(payload, metrics_path)
    if write_aliases:
        save_json(payload, results_root / "bert_test_metrics.json")

    save_comparison_row(
        results_root / "model_comparison.csv",
        {
            "Model": cfg.model_label,
            "Imbalance method": imbalance_name,
            "Accuracy": test_metrics["accuracy"],
            "Fraud Precision": test_metrics["fraud_precision"],
            "Fraud Recall": test_metrics["fraud_recall"],
            "Fraud F1": test_metrics["fraud_f1"],
            "Macro F1": test_metrics["macro_f1"],
            "ROC-AUC": test_metrics["roc_auc"],
            "PR-AUC": test_metrics["pr_auc"],
            "Inference time (s)": inference_time,
            "Notes": f"run={cfg.run_name}; thr={best_thr:.4f}; best_epoch={best_epoch}",
        },
    )
    plot_model_comparison(results_root / "model_comparison.csv", figures_root / "model_comparison.png")

    run_config = {
        "experiment": f"{cfg.model_label.lower()}_finetune",
        "paths": default_paths_dict(),
        "config": config_to_dict(cfg),
        "environment": collect_environment_info(),
        "best_checkpoint": str(best_dir),
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric,
        "threshold": best_thr,
        "token_length_stats": token_stats,
        "training_time_sec": total_timer.elapsed(),
        "imbalance_method": imbalance_name,
        "class_weights": class_weights.tolist(),
        "sample_counts": {
            k: {"n": len(v), "fraud": int((v[LABEL_COLUMN] == 1).sum())} for k, v in splits.items()
        },
        "train_oversample": {
            "factor": float(cfg.fraud_oversample_factor),
            "n_before": n_train_before,
            "n_after": n_train_after,
            "fraud_before": n_fraud_before,
            "fraud_after": n_fraud_after,
        },
        "class_weight_transform": cfg.class_weight_transform,
        "class_weight_max": cfg.class_weight_max,
        "threshold_step": cfg.threshold_step,
        "min_fraud_recall_for_threshold": cfg.min_fraud_recall_for_threshold,
        "results_dir": str(results_root),
        "figures_dir": str(figures_root),
    }
    save_json(run_config, results_root / f"run_config_{cfg.run_name}.json")
    if write_aliases:
        save_json(run_config, results_root / "run_config.json")

    # Persist threshold next to checkpoint for bert.py predict
    save_json(
        {
            "threshold": best_thr,
            "default_threshold": 0.5,
            "model_name": cfg.pretrained_model_name,
            "run_name": cfg.run_name,
            "imbalance_method": imbalance_name,
        },
        best_dir / "threshold.json",
    )

    logger.info("Test fraud_f1=%.4f fraud_recall=%.4f pr_auc=%.4f", test_metrics["fraud_f1"], test_metrics["fraud_recall"], test_metrics["pr_auc"])
    logger.info("Done. Artifacts under %s and %s", output_dir, results_root)
    return payload


# ==============================================================================
# CLI COMMANDS: evaluate / predict / export-val
# ==============================================================================

import argparse
import shutil
import traceback


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a saved checkpoint on the fixed test split."""
    ensure_directories()
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

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = BertForFraudClassification(PRETRAINED_MODEL_NAME)
    model.model = type(model.model).from_pretrained(ckpt)
    model = model.to(device)

    test_df = load_split("test")
    ds = dataframe_to_text_dataset(test_df, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda feats: collate_batch(feats, collator),
        num_workers=0,
    )

    _, y_true, y_prob, ids, titles, companies, idxs = evaluate_loader(
        model, loader, device, criterion=None, use_amp=device.type == "cuda"
    )
    metrics = binary_metrics(y_true, y_prob, threshold=threshold)
    metrics_05 = binary_metrics(y_true, y_prob, threshold=0.5)

    preds = build_predictions_frame(
        record_ids=ids,
        titles=titles,
        companies=companies,
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        model_name=f"BERT:{ckpt.name}",
        imbalance_strategy="checkpoint",
        row_indices=idxs,
    )
    preds.to_csv(RESULTS_DIR / "predictions_eval.csv", index=False)
    errors = build_error_analysis(preds, test_df["model_text"].tolist())
    errors.to_csv(RESULTS_DIR / "error_analysis_eval.csv", index=False)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        FIGURES_DIR / "confusion_matrix_eval.png",
        title="BERT eval Confusion Matrix",
    )
    plot_roc_pr(
        y_true,
        y_prob,
        FIGURES_DIR / "roc_curve_eval.png",
        FIGURES_DIR / "precision_recall_curve_eval.png",
    )

    payload = {
        "checkpoint": str(ckpt),
        "threshold": threshold,
        "test_metrics": metrics,
        "test_metrics_threshold_0_5": metrics_05,
    }
    save_json(payload, RESULTS_DIR / "bert_eval_test_metrics.json")
    logger.info(
        "Test fraud_f1=%.4f fraud_recall=%.4f pr_auc=%.4f accuracy=%.4f",
        metrics["fraud_f1"],
        metrics["fraud_recall"],
        metrics["pr_auc"],
        metrics["accuracy"],
    )
    print(f"Fraud F1: {metrics['fraud_f1']:.4f}")
    print(f"Fraud Recall: {metrics['fraud_recall']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Threshold: {threshold:.4f}")


def load_predictor(
    checkpoint_dir: Path,
    device: torch.device,
    threshold_override: Optional[float] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    wrapper = BertForFraudClassification(PRETRAINED_MODEL_NAME)
    wrapper.model = type(wrapper.model).from_pretrained(checkpoint_dir)
    wrapper = wrapper.to(device)
    wrapper.eval()

    thr_path = checkpoint_dir / "threshold.json"
    meta_path = checkpoint_dir / "train_meta.json"
    threshold = 0.5
    model_name = PRETRAINED_MODEL_NAME
    if thr_path.exists():
        thr_obj = load_json(thr_path)
        threshold = float(thr_obj.get("threshold", 0.5))
        model_name = thr_obj.get("model_name", model_name)
    if threshold_override is not None:
        threshold = float(threshold_override)
    meta = load_json(meta_path) if meta_path.exists() else {}
    return wrapper, tokenizer, threshold, model_name, meta


@torch.no_grad()
def predict_texts(
    texts: List[str],
    model: BertForFraudClassification,
    tokenizer,
    device: torch.device,
    threshold: float,
    max_length: int = 384,
) -> List[Dict[str, Any]]:
    model.eval()
    results = []
    for text in texts:
        cleaned = clean_text(text)
        enc = tokenizer(
            cleaned,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        fraud_p = float(softmax_fraud_proba(logits)[0].cpu().item())
        legit_p = 1.0 - fraud_p
        pred = 1 if fraud_p >= threshold else 0
        label = "Fraudulent" if pred == 1 else "Legitimate"
        results.append(
            {
                "predicted_label": label,
                "predicted_label_id": pred,
                "fraud_probability": fraud_p,
                "legitimate_probability": legit_p,
                "fraud_risk_score": fraud_p,
                "threshold": threshold,
            }
        )
    return results


def format_prediction(result: Dict[str, Any], model_name: str) -> str:
    return (
        f"Predicted label: {result['predicted_label']}\n"
        f"Fraud probability: {result['fraud_probability']:.4f}\n"
        f"Legitimate probability: {result['legitimate_probability']:.4f}\n"
        f"Fraud risk score: {result['fraud_risk_score']:.4f}\n"
        f"Threshold: {result['threshold']:.4f}\n"
        f"Model: {model_name}"
    )


def cmd_predict(args: argparse.Namespace) -> None:
    ensure_directories()
    logger = setup_logging("training.log")
    device = get_device(allow_cpu=args.allow_cpu, logger=logger)
    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. Train first with: python BERT/code/bert.py train"
        )

    model, tokenizer, threshold, model_name, _meta = load_predictor(
        ckpt, device, threshold_override=args.threshold
    )

    if args.text:
        result = predict_texts(
            [args.text], model, tokenizer, device, threshold, args.max_length
        )[0]
        print(format_prediction(result, model_name))
        return

    if args.csv:
        df = pd.read_csv(args.csv)
        texts = []
        for _, row in df.iterrows():
            if args.text_column in df.columns:
                texts.append(clean_text(row[args.text_column]))
            else:
                texts.append(extract_text_from_row(row.to_dict()))
        results = predict_texts(
            texts, model, tokenizer, device, threshold, args.max_length
        )
        out = df.copy()
        out["predicted_label"] = [r["predicted_label"] for r in results]
        out["fraud_probability"] = [r["fraud_probability"] for r in results]
        out["legitimate_probability"] = [r["legitimate_probability"] for r in results]
        out["fraud_risk_score"] = [r["fraud_risk_score"] for r in results]
        out["threshold"] = threshold
        out_path = (
            Path(args.output_csv)
            if args.output_csv
            else Path(args.csv).with_name(Path(args.csv).stem + "_bert_predictions.csv")
        )
        out.to_csv(out_path, index=False)
        print(f"Wrote predictions to {out_path}")
        return

    raise SystemExit("predict requires --text or --csv")


def cmd_export_val(args: argparse.Namespace) -> None:
    """Export BERT validation scores for the FP-gate ensemble."""
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

    if ENSEMBLE_VAL_COPY.parent.exists():
        shutil.copy2(out_path, ENSEMBLE_VAL_COPY)
        logger.info("Also refreshed ensemble copy: %s", ENSEMBLE_VAL_COPY)


def cmd_train(args: argparse.Namespace) -> None:
    run_name = args.run_name
    if run_name is None:
        run_name = "bert_cw_improved" if args.use_class_weights else "bert_no_class_weight"
    cfg = BertFinetuneConfig(
        pretrained_model_name=args.pretrained_model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        dropout=args.dropout,
        random_seed=args.random_seed,
        fp16=args.fp16,
        use_class_weights=args.use_class_weights,
        class_weight_transform=args.class_weight_transform,
        class_weight_max=args.class_weight_max,
        fraud_oversample_factor=args.fraud_oversample_factor,
        gradient_checkpointing=args.gradient_checkpointing,
        optimize_threshold=args.optimize_threshold,
        threshold_selection_mode=args.threshold_selection_mode,
        threshold_f_beta=args.threshold_f_beta,
        min_fraud_recall_for_threshold=args.min_fraud_recall_for_threshold,
        threshold_step=args.threshold_step,
        output_dir=args.output_dir,
        run_name=run_name,
    )
    try:
        run_training(cfg)
    except SystemExit:
        raise
    except Exception:
        logger = setup_logging("training.log")
        logger.error("Training failed:\n%s", traceback.format_exc())
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BERT fraudulent job-ad detection (train / evaluate / predict / export-val)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fine-tune BERT on fixed train/val/test splits")
    p_train.add_argument("--pretrained_model_name", default=PRETRAINED_MODEL_NAME)
    p_train.add_argument("--learning_rate", type=float, default=2e-5)
    p_train.add_argument("--weight_decay", type=float, default=0.01)
    p_train.add_argument("--num_train_epochs", type=int, default=5)
    p_train.add_argument("--train_batch_size", type=int, default=4)
    p_train.add_argument("--eval_batch_size", type=int, default=8)
    p_train.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p_train.add_argument("--max_length", type=int, default=MAX_LENGTH)
    p_train.add_argument("--warmup_ratio", type=float, default=0.1)
    p_train.add_argument("--early_stopping_patience", type=int, default=2)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--random_seed", type=int, default=RANDOM_SEED)
    p_train.add_argument("--fp16", type=parse_bool, default=True)
    p_train.add_argument("--use_class_weights", type=parse_bool, default=True)
    p_train.add_argument(
        "--class_weight_transform",
        type=str,
        default="sqrt_clip",
        choices=["none", "sqrt", "clip", "sqrt_clip"],
    )
    p_train.add_argument("--class_weight_max", type=float, default=5.0)
    p_train.add_argument("--fraud_oversample_factor", type=float, default=3.0)
    p_train.add_argument("--gradient_checkpointing", type=parse_bool, default=False)
    p_train.add_argument("--optimize_threshold", type=parse_bool, default=True)
    p_train.add_argument(
        "--threshold_selection_mode",
        type=str,
        default="max_fbeta",
        choices=[
            "max_fraud_f1",
            "max_fbeta",
            "near_max_f1_prefer_precision",
            "min_recall_then_f1",
            "min_recall_then_precision",
        ],
    )
    p_train.add_argument("--threshold_f_beta", type=float, default=0.5)
    p_train.add_argument("--min_fraud_recall_for_threshold", type=float, default=0.85)
    p_train.add_argument("--threshold_step", type=float, default=0.01)
    p_train.add_argument("--run_name", type=str, default=None)
    p_train.add_argument("--output_dir", type=str, default=str(BERT_FINETUNED_DIR))
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate checkpoint on fixed test split")
    p_eval.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(BERT_FINETUNED_DIR / "best"),
    )
    p_eval.add_argument("--threshold", type=float, default=None)
    p_eval.add_argument("--eval_batch_size", type=int, default=8)
    p_eval.add_argument("--max_length", type=int, default=512)
    p_eval.add_argument("--allow_cpu", action="store_true")
    p_eval.add_argument("--random_seed", type=int, default=42)
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Predict on a single text or CSV of ads")
    p_pred.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(BERT_FINETUNED_DIR / "best"),
    )
    p_pred.add_argument("--text", type=str, default=None, help="Single job-ad text")
    p_pred.add_argument("--csv", type=str, default=None, help="Optional CSV of ads")
    p_pred.add_argument("--text_column", type=str, default="combined_text")
    p_pred.add_argument("--output_csv", type=str, default=None)
    p_pred.add_argument("--threshold", type=float, default=None)
    p_pred.add_argument("--max_length", type=int, default=MAX_LENGTH)
    p_pred.add_argument("--allow_cpu", action="store_true", default=True)
    p_pred.set_defaults(func=cmd_predict)

    p_exp = sub.add_parser(
        "export-val",
        help="Export validation fraud scores for ensemble FP-gate",
    )
    p_exp.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(BERT_FINETUNED_DIR / "best"),
    )
    p_exp.add_argument("--eval_batch_size", type=int, default=16)
    p_exp.add_argument("--max_length", type=int, default=512)
    p_exp.add_argument("--allow_cpu", action="store_true")
    p_exp.add_argument("--random_seed", type=int, default=42)
    p_exp.set_defaults(func=cmd_export_val)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
