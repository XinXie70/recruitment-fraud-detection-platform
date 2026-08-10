"""Aggregate LR / DNN test metrics into a Fraud-F1 ranked table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = {
    "LR": ROOT / "LR" / "result" / "test_metrics.json",
    "DNN": ROOT / "DNN" / "result" / "test_metrics.json",
}


def main() -> None:
    rows = []
    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics for {name}: {path}")
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": name,
                "fraud_f1": m["fraud_f1"],
                "fraud_recall": m["fraud_recall"],
                "fraud_precision": m["fraud_precision"],
            }
        )

    table = pd.DataFrame(rows).sort_values("fraud_f1", ascending=False).reset_index(drop=True)
    out_csv = ROOT / "model_comparison.csv"
    table.to_csv(out_csv, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
