"""Shared helpers for Imbalance Handling Strategies for BERT.

Reuses the sibling BERT training stack under ../BERT/code with default
BertFinetuneConfig hyperparameters. Artifacts stay under this experiment root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = EXPERIMENT_ROOT / "code"
WEIGHT_DIR = EXPERIMENT_ROOT / "weight"
RESULT_DIR = EXPERIMENT_ROOT / "result"
COMPARISON_CSV = RESULT_DIR / "comparison_test.csv"

# Keep only these evaluation figures under result/<strategy>/figures/.
KEEP_FIGURES = {
    "class_weight": {
        "confusion_matrix_class_weight.png",
        "precision_recall_curve_class_weight.png",
        "roc_curve_class_weight.png",
    },
    "smote": {
        "confusion_matrix_smote.png",
        "precision_recall_curve_smote.png",
        "roc_curve_smote.png",
    },
}


SPRINT3 = EXPERIMENT_ROOT.parent
BERT_CODE = SPRINT3 / "BERT" / "code"
DATA_SPLITS = SPRINT3 / "data" / "splits"

if str(BERT_CODE) not in sys.path:
    sys.path.insert(0, str(BERT_CODE))

from config import BertFinetuneConfig  # noqa: E402
from data_metrics import load_all_splits  # noqa: E402
from training import run_training  # noqa: E402


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_splits() -> Dict[str, pd.DataFrame]:
    """Load fixed train/validation/test from sprint3/data/splits."""
    if not DATA_SPLITS.exists():
        raise FileNotFoundError(f"Missing data splits directory: {DATA_SPLITS}")
    return load_all_splits(
        train_path=DATA_SPLITS / "train.csv.gz",
        validation_path=DATA_SPLITS / "validation.csv.gz",
        test_path=DATA_SPLITS / "test.csv.gz",
    )


def default_bert_config(**overrides: Any) -> BertFinetuneConfig:
    """Default BERT fine-tune hyperparameters (from BertFinetuneConfig)."""
    cfg = BertFinetuneConfig()
    # This experiment does not keep error_analysis_*.csv artifacts.
    cfg.write_error_analysis = False
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown BertFinetuneConfig field: {key}")
        setattr(cfg, key, value)
    return cfg


def prune_figures(strategy_name: str, figures_root: Path) -> None:
    """Delete figure files not in the keep-list for this strategy."""
    keep = KEEP_FIGURES.get(strategy_name)
    if keep is None:
        raise ValueError(f"No keep-list configured for strategy {strategy_name!r}")
    if not figures_root.exists():
        return
    removed = []
    for path in figures_root.iterdir():
        if not path.is_file():
            continue
        if path.name not in keep:
            path.unlink()
            removed.append(path.name)
    missing = sorted(keep - {p.name for p in figures_root.iterdir() if p.is_file()})
    print(
        f"Figures kept for {strategy_name}: {sorted(keep)}",
        flush=True,
    )
    if removed:
        print(f"Figures removed: {removed}", flush=True)
    if missing:
        print(f"WARNING: expected figures missing: {missing}", flush=True)


def run_experiment(
    *,
    strategy_name: str,
    cfg: BertFinetuneConfig,
    splits: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Fine-tune BERT and write weights/metrics under this experiment root."""
    weight_root = WEIGHT_DIR / strategy_name
    result_root = RESULT_DIR / strategy_name
    figures_root = result_root / "figures"
    ensure_dirs(weight_root, result_root, figures_root)

    cfg.output_dir = str(weight_root)
    cfg.run_name = strategy_name
    cfg.write_canonical_aliases = False

    print(f"=== Strategy: {strategy_name} ===", flush=True)
    print(f"Weights -> {weight_root}", flush=True)
    print(f"Results -> {result_root}", flush=True)
    print(
        f"Splits: train={len(splits['train'])} "
        f"val={len(splits['validation'])} test={len(splits['test'])}",
        flush=True,
    )

    payload = run_training(
        cfg,
        splits=splits,
        results_dir=result_root,
        figures_dir=figures_root,
        log_name=f"{strategy_name}.log",
    )
    prune_figures(strategy_name, figures_root)
    # Ensure error-analysis CSVs are never retained for this experiment.
    for leftover in result_root.glob("error_analysis*.csv"):
        leftover.unlink(missing_ok=True)
    test_metrics = payload.get("test_metrics") or {}
    row = {
        "strategy": strategy_name,
        "model": cfg.model_label,
        "Accuracy": float(test_metrics.get("accuracy", float("nan"))),
        "Fraud Precision": float(test_metrics.get("fraud_precision", float("nan"))),
        "Fraud Recall": float(test_metrics.get("fraud_recall", float("nan"))),
        "Fraud F1": float(test_metrics.get("fraud_f1", float("nan"))),
        "threshold": float(payload.get("threshold", test_metrics.get("threshold", 0.5))),
        "weight_dir": str(weight_root / "best"),
        "metrics_json": str(result_root / f"test_metrics_{strategy_name}.json"),
    }
    update_comparison(row)

    slim = {
        "strategy": strategy_name,
        "Accuracy": row["Accuracy"],
        "Fraud Precision": row["Fraud Precision"],
        "Fraud Recall": row["Fraud Recall"],
        "Fraud F1": row["Fraud F1"],
        "threshold": row["threshold"],
        "weight_dir": row["weight_dir"],
    }
    (result_root / "test_summary.json").write_text(
        json.dumps(slim, indent=2), encoding="utf-8"
    )
    print(json.dumps(slim, indent=2), flush=True)
    return payload


def update_comparison(row: Dict[str, Any]) -> None:
    """Keep a test-only comparison table (Accuracy / Fraud P / R / F1)."""
    ensure_dirs(RESULT_DIR)
    cols = ["strategy", "Accuracy", "Fraud Precision", "Fraud Recall", "Fraud F1"]
    slim = {k: row[k] for k in cols}
    if COMPARISON_CSV.exists():
        table = pd.read_csv(COMPARISON_CSV)
    else:
        table = pd.DataFrame(columns=cols)
    for col in cols:
        if col not in table.columns:
            table[col] = pd.NA
    mask = table["strategy"] == slim["strategy"]
    if mask.any():
        for key, value in slim.items():
            table.loc[mask, key] = value
    else:
        table = pd.concat([table, pd.DataFrame([slim])], ignore_index=True)
    table = table[cols]
    table.to_csv(COMPARISON_CSV, index=False)
    print(f"Updated comparison: {COMPARISON_CSV}", flush=True)
    print(table.to_string(index=False), flush=True)


def apply_smote_text_augmentation(
    train_df: pd.DataFrame,
    *,
    text_column: str = "model_text",
    label_column: str = "label",
    seed: int = 42,
    max_features: int = 8000,
    k_neighbors: int = 5,
) -> pd.DataFrame:
    """Balance the train split with SMOTE in TF-IDF space, then map back to text.

    Synthetic TF-IDF points are replaced by their nearest original training
    document (standard text-SMOTE approximation so BERT still receives strings).
    Validation/test must never be passed here.
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    if text_column not in train_df.columns:
        raise ValueError(f"train_df missing text column {text_column!r}")
    if label_column not in train_df.columns:
        raise ValueError(f"train_df missing label column {label_column!r}")

    base = train_df.reset_index(drop=True).copy()
    texts = base[text_column].fillna("").astype(str)
    y = base[label_column].astype(int).to_numpy()
    n_fraud = int((y == 1).sum())
    if n_fraud <= k_neighbors:
        raise ValueError(
            f"Not enough fraud samples for SMOTE k_neighbors={k_neighbors}: {n_fraud}"
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    x_tfidf = vectorizer.fit_transform(texts)
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    x_res, y_res = smote.fit_resample(x_tfidf, y)

    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(x_tfidf)
    neighbor_idx = nn.kneighbors(x_res, return_distance=False).reshape(-1)

    rows = []
    for i, src_i in enumerate(neighbor_idx.tolist()):
        row = base.iloc[int(src_i)].copy()
        row[label_column] = int(y_res[i])
        row["smote_source_index"] = int(src_i)
        row["smote_synthetic"] = int(i >= len(base))
        rows.append(row)

    out = pd.DataFrame(rows).reset_index(drop=True)
    # Keep a stable row_index for the downstream dataset helper.
    out["row_index"] = range(len(out))
    print(
        f"SMOTE train rows: {len(base)} -> {len(out)} "
        f"(fraud {int((base[label_column] == 1).sum())} -> {int((out[label_column] == 1).sum())})",
        flush=True,
    )
    return out
