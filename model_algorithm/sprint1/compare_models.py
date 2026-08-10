#Compare models for EMSCAD data.
from __future__ import annotations
import json, pandas as pd
from pathlib import Path
PROJECT_PATH = Path(__file__).resolve().parent
MODEL_FILES = {
    "LR": PROJECT_PATH / "LR" / "result" / "test_metrics.json",
    "DNN": PROJECT_PATH / "DNN" / "result" / "test_metrics.json",
}
def read_metrics(model_name: str, file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing metrics for {model_name}: {file_path}")
    metrics = json.loads(file_path.read_text(encoding="utf-8"))
    return {
        "model": model_name,
        "fraud_f1": metrics["fraud_f1"],
        "fraud_recall": metrics["fraud_recall"],
        "fraud_precision": metrics["fraud_precision"],
    }
def build_comparison_table() -> pd.DataFrame:
    rows = [read_metrics(model_name, file_path) for model_name, file_path in MODEL_FILES.items()]
    return pd.DataFrame(rows).sort_values("fraud_f1", ascending=False).reset_index(drop=True)
def save_result(table: pd.DataFrame) -> Path:
    output_file = PROJECT_PATH / "model_comparison.csv"
    table.to_csv(output_file, index=False)
    return output_file
def main() -> None:
    comparison_table = build_comparison_table()
    output_file = save_result(comparison_table)
    print(comparison_table.to_string(index=False))
    print(f"\nSaved: {output_file}")
if __name__ == "__main__":
    main()