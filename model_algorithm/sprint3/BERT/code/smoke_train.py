"""Smoke test for BERT train pipeline.

Runs a 1-epoch mini fine-tune on tiny subsets so the full train path
(data -> model -> checkpoint -> metrics) is exercised without a long job.

Usage (from BERT/code, with CUDA env):
  python smoke_train.py
"""

from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import (  # noqa: E402
    LABEL_COLUMN,
    MODEL_LABEL,
    PROJECT_ROOT,
    BertFinetuneConfig,
)
from data_metrics import load_all_splits  # noqa: E402
from training import run_training  # noqa: E402


def _mini_split(df, n: int, seed: int = 42):
    """Keep both classes when possible for a tiny balanced-ish sample."""
    fraud = df[df[LABEL_COLUMN] == 1]
    legit = df[df[LABEL_COLUMN] == 0]
    n_fraud = max(1, min(len(fraud), max(1, n // 4)))
    n_legit = max(1, min(len(legit), n - n_fraud))
    parts = []
    if len(fraud):
        parts.append(fraud.sample(n=n_fraud, random_state=seed))
    if len(legit):
        parts.append(legit.sample(n=n_legit, random_state=seed))
    out = (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    if len(out) < 2:
        raise RuntimeError(f"mini split too small: {len(out)}")
    return out


def main() -> int:
    smoke_root = PROJECT_ROOT / "test" / "smoke_train"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    weights_dir = smoke_root / "weight"
    results_dir = smoke_root / "results"
    figures_dir = results_dir / "figures"
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== BERT train smoke test ===", flush=True)
    print(f"Smoke output: {smoke_root}", flush=True)

    full = load_all_splits()
    splits = {
        "train": _mini_split(full["train"], n=48, seed=42),
        "validation": _mini_split(full["validation"], n=16, seed=43),
        "test": _mini_split(full["test"], n=16, seed=44),
    }
    for name, df in splits.items():
        print(
            f"  {name}: n={len(df)} fraud={int((df[LABEL_COLUMN] == 1).sum())}",
            flush=True,
        )

    cfg = BertFinetuneConfig(
        pretrained_model_name="bert-base-uncased",
        learning_rate=5e-5,
        num_train_epochs=1,
        train_batch_size=2,
        eval_batch_size=4,
        gradient_accumulation_steps=1,
        max_length=64,
        warmup_ratio=0.0,
        early_stopping_patience=1,
        fp16=True,
        use_class_weights=True,
        class_weight_transform="none",
        fraud_oversample_factor=1.0,
        optimize_threshold=True,
        threshold_selection_mode="max_fraud_f1",
        min_fraud_recall_for_threshold=None,
        output_dir=str(weights_dir),
        run_name="smoke",
        model_label=MODEL_LABEL,
        write_canonical_aliases=True,
    )

    try:
        payload = run_training(
            cfg,
            splits=splits,
            results_dir=results_dir,
            figures_dir=figures_dir,
            log_name="smoke_train.log",
        )
    except SystemExit as exc:
        print(f"FAIL: training aborted (SystemExit={exc.code})", flush=True)
        return 1
    except Exception:
        print("FAIL: training raised an exception:", flush=True)
        traceback.print_exc()
        return 1

    best_dir = weights_dir / "best"
    required = [
        best_dir / "config.json",
        best_dir / "threshold.json",
        results_dir / "test_metrics_smoke.json",
        results_dir / "predictions_smoke.csv",
        results_dir / "training_history_smoke.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("FAIL: missing artifacts:", flush=True)
        for p in missing:
            print(f"  - {p}", flush=True)
        return 1

    fraud_f1 = float(payload.get("test_metrics", {}).get("fraud_f1", float("nan")))
    print("PASS: train smoke completed", flush=True)
    print(f"  checkpoint: {best_dir}", flush=True)
    print(f"  test fraud_f1 (tiny split): {fraud_f1:.4f}", flush=True)
    print(f"  artifacts under: {smoke_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
