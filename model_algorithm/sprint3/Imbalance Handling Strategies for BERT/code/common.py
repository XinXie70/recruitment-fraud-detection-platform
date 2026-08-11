#Common functions for Imbalance Handling Strategies for BERT
from __future__ import annotations
import json, sys, pandas as pd
from pathlib import Path
from typing import Any, Dict
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

EXPERIMENT_PATH = Path(__file__).resolve().parents[1]
CODE_PATH = EXPERIMENT_PATH / "code"
MODEL_PATH = EXPERIMENT_PATH / "weight"
OUTPUT_PATH = EXPERIMENT_PATH / "result"
COMPARISON_PATH = OUTPUT_PATH / "comparison_test.csv"
SPRINT3_PATH = EXPERIMENT_PATH.parent
BERT_CODE_PATH = SPRINT3_PATH / "BERT" / "code"
SPLIT_PATH = SPRINT3_PATH / "data" / "splits"
KEEP_FIGURES = {
    "class_weight": {"confusion_matrix_class_weight.png", "precision_recall_curve_class_weight.png", "roc_curve_class_weight.png"},
    "smote": {"confusion_matrix_smote.png", "precision_recall_curve_smote.png", "roc_curve_smote.png"},
}
if str(BERT_CODE_PATH) not in sys.path:
    sys.path.insert(0, str(BERT_CODE_PATH))

from config import BertFinetuneConfig
from data_metrics import load_all_splits
from training import run_training

def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def read_splits() -> Dict[str, pd.DataFrame]:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Missing data splits directory: {SPLIT_PATH}")
    return load_all_splits(train_path=SPLIT_PATH / "train.csv.gz", validation_path=SPLIT_PATH / "validation.csv.gz", test_path=SPLIT_PATH / "test.csv.gz")


def create_bert_config(**overrides: Any) -> BertFinetuneConfig:
    config = BertFinetuneConfig()
    config.write_error_analysis = False
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"Unknown BertFinetuneConfig field: {key}")
        setattr(config, key, value)
    return config


def clean_figures(strategy_name: str, figures_path: Path) -> None:
    keep_files = KEEP_FIGURES.get(strategy_name)
    if keep_files is None:
        raise ValueError(f"No keep-list configured for strategy {strategy_name!r}")
    if not figures_path.exists():
        return
    removed_files = []
    for file_path in figures_path.iterdir():
        if file_path.is_file() and file_path.name not in keep_files:
            file_path.unlink()
            removed_files.append(file_path.name)
    existing_files = {file_path.name for file_path in figures_path.iterdir() if file_path.is_file()}
    missing_files = sorted(keep_files - existing_files)
    print(f"Figures kept for {strategy_name}: {sorted(keep_files)}", flush=True)
    if removed_files:
        print(f"Figures removed: {removed_files}", flush=True)
    if missing_files:
        print(f"WARNING: expected figures missing: {missing_files}", flush=True)

def update_comparison(row: Dict[str, Any]) -> None:
    ensure_directories(OUTPUT_PATH)
    columns = ["strategy", "Accuracy", "Fraud Precision", "Fraud Recall", "Fraud F1"]
    comparison_row = {column: row[column] for column in columns}
    if COMPARISON_PATH.exists():
        table = pd.read_csv(COMPARISON_PATH)
    else:
        table = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in table.columns:
            table[column] = pd.NA
    mask = table["strategy"] == comparison_row["strategy"]
    if mask.any():
        for key, value in comparison_row.items():
            table.loc[mask, key] = value
    else:
        table = pd.concat([table, pd.DataFrame([comparison_row])], ignore_index=True)
    table = table[columns]
    table.to_csv(COMPARISON_PATH, index=False)
    print(f"Updated comparison: {COMPARISON_PATH}", flush=True)
    print(table.to_string(index=False), flush=True)

def run_experiment(strategy_name: str, config: BertFinetuneConfig, splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    weight_path = MODEL_PATH / strategy_name
    result_path = OUTPUT_PATH / strategy_name
    figures_path = result_path / "figures"
    ensure_directories(weight_path, result_path, figures_path)
    config.output_dir = str(weight_path)
    config.run_name = strategy_name
    config.write_canonical_aliases = False
    print(f"=== Strategy: {strategy_name} ===", flush=True)
    print(f"Weights -> {weight_path}", flush=True)
    print(f"Results -> {result_path}", flush=True)
    print(f"Splits: train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])}", flush=True)
    payload = run_training(config, splits=splits, results_dir=result_path, figures_dir=figures_path, log_name=f"{strategy_name}.log")
    clean_figures(strategy_name, figures_path)
    for file_path in result_path.glob("error_analysis*.csv"):
        file_path.unlink(missing_ok=True)

    test_metrics = payload.get("test_metrics") or {}
    comparison_row = {
        "strategy": strategy_name,
        "model": config.model_label,
        "Accuracy": float(test_metrics.get("accuracy", float("nan"))),
        "Fraud Precision": float(test_metrics.get("fraud_precision", float("nan"))),
        "Fraud Recall": float(test_metrics.get("fraud_recall", float("nan"))),
        "Fraud F1": float(test_metrics.get("fraud_f1", float("nan"))),
        "threshold": float(payload.get("threshold", test_metrics.get("threshold", 0.5))),
        "weight_dir": str(weight_path / "best"),
        "metrics_json": str(result_path / f"test_metrics_{strategy_name}.json"),
    }

    update_comparison(comparison_row)
    summary = {
        "strategy": strategy_name,
        "Accuracy": comparison_row["Accuracy"],
        "Fraud Precision": comparison_row["Fraud Precision"],
        "Fraud Recall": comparison_row["Fraud Recall"],
        "Fraud F1": comparison_row["Fraud F1"],
        "threshold": comparison_row["threshold"],
        "weight_dir": comparison_row["weight_dir"],
    }
    (result_path / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return payload


def apply_smote_text_augmentation(train_data: pd.DataFrame, text_column: str = "model_text", label_column: str = "label", seed: int = 42, max_features: int = 8000, k_neighbors: int = 5) -> pd.DataFrame:
    if text_column not in train_data.columns:
        raise ValueError(f"train_data missing text column {text_column!r}")
    if label_column not in train_data.columns:
        raise ValueError(f"train_data missing label column {label_column!r}")
    base_data = train_data.reset_index(drop=True).copy()
    texts = base_data[text_column].fillna("").astype(str)
    labels = base_data[label_column].astype(int).to_numpy()
    fraud_count = int((labels == 1).sum())
    if fraud_count <= k_neighbors:
        raise ValueError(f"Not enough fraud samples for SMOTE k_neighbors={k_neighbors}: {fraud_count}")
    vectorizer = TfidfVectorizer(lowercase=True, min_df=2, max_df=0.98, max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    tfidf_features = vectorizer.fit_transform(texts)
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    resampled_features, resampled_labels = smote.fit_resample(tfidf_features, labels)
    nearest_neighbors = NearestNeighbors(n_neighbors=1, metric="cosine").fit(tfidf_features)
    neighbor_indices = nearest_neighbors.kneighbors(resampled_features, return_distance=False).reshape(-1)
    rows = []
    for index, source_index in enumerate(neighbor_indices.tolist()):
        row = base_data.iloc[int(source_index)].copy()
        row[label_column] = int(resampled_labels[index])
        row["smote_source_index"] = int(source_index)
        row["smote_synthetic"] = int(index >= len(base_data))
        rows.append(row)

    output_data = pd.DataFrame(rows).reset_index(drop=True)
    output_data["row_index"] = range(len(output_data))
    original_fraud = int((base_data[label_column] == 1).sum())
    resampled_fraud = int((output_data[label_column] == 1).sum())
    print(f"SMOTE train rows: {len(base_data)} -> {len(output_data)} (fraud {original_fraud} -> {resampled_fraud})", flush=True)
    return output_data