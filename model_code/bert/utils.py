"""Shared utilities: seeding, CUDA checks, logging, JSON/CSV helpers."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from config import LOGS_DIR, ensure_directories

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
            "  E:\\path\\to\\cuda_environment\\python.exe train_bert.py\n"
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
