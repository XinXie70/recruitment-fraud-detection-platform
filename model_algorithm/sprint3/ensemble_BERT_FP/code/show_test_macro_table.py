#Show test macro table for BERT + FP-gate for EMSCAD data
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any, Iterable, Optional
CODE_PATH = Path(__file__).resolve().parent
PROJECT_PATH = CODE_PATH.parent
SPRINT3_PATH = PROJECT_PATH.parent
LR_METRICS_PATH = SPRINT3_PATH / "LR" / "results" / "test_metrics.json"
ENSEMBLE_METRICS_PATH = PROJECT_PATH / "results" / "test_metrics.json"
BERT_METRICS_FILES = (
    SPRINT3_PATH / "BERT" / "results" / "metrics_test.json",
    SPRINT3_PATH / "BERT" / "results" / "bert-paper-protocol_test_metrics_bert_paper_protocol_maxlen512.json",
)
if str(CODE_PATH) not in sys.path:
    sys.path.insert(0, str(CODE_PATH))
def find_existing_file(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None
def read_json(file_path: Path) -> dict[str, Any]:
    return json.loads(file_path.read_text(encoding="utf-8"))
def read_bert_macro_metrics(file_path: Path) -> dict[str, float]:
    data = read_json(file_path)
    if "metrics" in data and isinstance(data["metrics"], dict):
        metrics = data["metrics"]
        report = metrics.get("classification_report") or {}
        macro = report.get("macro avg") or {}
        return {
            "macro_precision": float(data.get("macro_precision", macro.get("precision", metrics.get("macro_precision", 0.0)))),
            "macro_recall": float(data.get("macro_recall", macro.get("recall", metrics.get("macro_recall", 0.0)))),
            "macro_f1": float(data.get("macro_f1", metrics.get("macro_f1", macro.get("f1-score", 0.0)))),
        }

    if "test_metrics" in data:
        metrics = data["test_metrics"].get("threshold_optimized") or data["test_metrics"]
        report = metrics.get("classification_report") or {}
        macro = report.get("macro avg") or {}
        return {
            "macro_precision": float(macro.get("precision", 0.0)),
            "macro_recall": float(macro.get("recall", 0.0)),
            "macro_f1": float(metrics.get("macro_f1", macro.get("f1-score", 0.0))),
        }
    raise ValueError(f"Unrecognized BERT metrics schema: {file_path}")

def format_metric(value: float) -> str:
    return f"{value:.4f}"

def build_rows(lr_metrics: dict, bert_metrics: dict, ensemble_metrics: dict) -> list[tuple]:
    return [
        ("LR", lr_metrics["macro_precision"], lr_metrics["macro_recall"], lr_metrics["macro_f1"]),
        ("Optimized BERT", bert_metrics["macro_precision"], bert_metrics["macro_recall"], bert_metrics["macro_f1"]),
        ("FP-gate Ensemble", ensemble_metrics["macro_precision"], ensemble_metrics["macro_recall"], ensemble_metrics["macro_f1"]),
    ]

def print_comparison(rows: list[tuple]) -> None:
    title = "Test 三方对比（Macro）"
    headers = ("Model", "Macro P", "Macro R", "Macro F1")
    column_widths = [18, 10, 10, 10]
    line_width = sum(column_widths) + 3
    print(title)
    print("=" * line_width)
    print(f"{headers[0]:<{column_widths[0]}}{headers[1]:>{column_widths[1]}}{headers[2]:>{column_widths[2]}}{headers[3]:>{column_widths[3]}}")
    print("-" * line_width)
    for model_name, precision, recall, f1 in rows:
        marker = " *" if model_name == "FP-gate Ensemble" else ""
        print(f"{model_name:<{column_widths[0]}}{format_metric(precision):>{column_widths[1]}}{format_metric(recall):>{column_widths[2]}}{format_metric(f1):>{column_widths[3]}}{marker}")
    print("=" * line_width)
    print("* = FP-gate Ensemble (best across Macro P / R / F1 in current freeze)")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    bert_metrics_path = find_existing_file(BERT_METRICS_FILES)
    missing_files = [path for path in (LR_METRICS_PATH, ENSEMBLE_METRICS_PATH) if not path.exists()]

    if bert_metrics_path is None:
        missing_files.append(Path("BERT/results/metrics_test.json"))
    if missing_files:
        print("Missing required metrics files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("Run the ensemble first: python run_fp_gate_ensemble.py")
        return 1

    lr_metrics = read_json(LR_METRICS_PATH)
    bert_metrics = read_bert_macro_metrics(bert_metrics_path)
    ensemble_metrics = read_json(ENSEMBLE_METRICS_PATH)
    comparison_rows = build_rows(lr_metrics, bert_metrics, ensemble_metrics)
    print_comparison(comparison_rows)
    csv_path = PROJECT_PATH / "test_macro_comparison.csv"
    csv_path.write_text(
        "Model,Macro P,Macro R,Macro F1\n"
        + "".join(
            f"{model_name},{precision:.4f},{recall:.4f},{f1:.4f}\n"
            for model_name, precision, recall, f1 in comparison_rows
        ),
        encoding="utf-8",
    )
    print(f"Saved CSV: {csv_path}")
    print(f"Sources:\n  LR:       {LR_METRICS_PATH}\n  BERT:     {bert_metrics_path}\n  Ensemble: {ENSEMBLE_METRICS_PATH}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())