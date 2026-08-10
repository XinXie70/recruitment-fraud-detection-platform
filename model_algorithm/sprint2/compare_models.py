#Compare models for EMSCAD data
from __future__ import annotations
import json, pandas as pd
from pathlib import Path
PROJECT_PATH = Path(__file__).resolve().parent
SPRINT1_PATH = PROJECT_PATH.parent / "sprint1"
MODEL_FILES = [
    ("LR", "sprint1", SPRINT1_PATH / "LR" / "result" / "test_metrics.json"),
    ("DNN", "sprint1", SPRINT1_PATH / "DNN" / "result" / "test_metrics.json"),
    ("SVM", "sprint2", PROJECT_PATH / "SVM" / "result" / "test_metrics.json"),
    ("XGBOOST", "sprint2", PROJECT_PATH / "XGBOOST" / "result" / "test_metrics.json"),
    ("RNN", "sprint2", PROJECT_PATH / "RNN" / "result" / "test_metrics.json"),
    ("Bi-lstm", "sprint2", PROJECT_PATH / "Bi-lstm" / "result" / "test_metrics.json"),
]
def read_metrics(model_name: str, sprint_name: str, file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing metrics for {model_name}: {file_path}")
    metrics = json.loads(file_path.read_text(encoding="utf-8"))
    return {
        "sprint": sprint_name,
        "model": model_name,
        "fraud_f1": metrics["fraud_f1"],
        "fraud_recall": metrics["fraud_recall"],
        "fraud_precision": metrics["fraud_precision"],
    }
def build_comparison_table() -> pd.DataFrame:
    rows = [read_metrics(model_name, sprint_name, file_path) for model_name, sprint_name, file_path in MODEL_FILES]
    return pd.DataFrame(rows).sort_values("fraud_f1", ascending=False).reset_index(drop=True)
def save_results(table: pd.DataFrame) -> tuple[Path, Path]:
    sprint2_file = PROJECT_PATH / "model_comparison.csv"
    combined_file = PROJECT_PATH.parent / "sprint1_sprint2_model_comparison.csv"
    table.to_csv(sprint2_file, index=False)
    table.to_csv(combined_file, index=False)
    return sprint2_file, combined_file

def main() -> None:
    comparison_table = build_comparison_table()
    sprint2_file, combined_file = save_results(comparison_table)
    print(comparison_table.to_string(index=False))
    print(f"\nSaved: {sprint2_file}")
    print(f"Saved: {combined_file}")
if __name__ == "__main__":
    main()