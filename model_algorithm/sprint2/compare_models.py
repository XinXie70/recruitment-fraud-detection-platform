"""Aggregate sprint1 + sprint2 model test metrics, ranked by Fraud F1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SPRINT1 = ROOT.parent / "sprint1"

MODELS = [
    ("LR", "sprint1", SPRINT1 / "LR" / "result" / "test_metrics.json"),
    ("DNN", "sprint1", SPRINT1 / "DNN" / "result" / "test_metrics.json"),
    ("SVM", "sprint2", ROOT / "SVM" / "result" / "test_metrics.json"),
    ("XGBOOST", "sprint2", ROOT / "XGBOOST" / "result" / "test_metrics.json"),
    ("RNN", "sprint2", ROOT / "RNN" / "result" / "test_metrics.json"),
    ("Bi-lstm", "sprint2", ROOT / "Bi-lstm" / "result" / "test_metrics.json"),
]


def main() -> None:
    rows = []
    for name, sprint, path in MODELS:
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics for {name}: {path}")
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "sprint": sprint,
                "model": name,
                "fraud_f1": m["fraud_f1"],
                "fraud_recall": m["fraud_recall"],
                "fraud_precision": m["fraud_precision"],
            }
        )

    table = pd.DataFrame(rows).sort_values("fraud_f1", ascending=False).reset_index(drop=True)
    out_csv = ROOT / "model_comparison.csv"
    table.to_csv(out_csv, index=False)
    # Also refresh a combined table under model_algorithm for convenience.
    combined = ROOT.parent / "sprint1_sprint2_model_comparison.csv"
    table.to_csv(combined, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {combined}")


if __name__ == "__main__":
    main()
