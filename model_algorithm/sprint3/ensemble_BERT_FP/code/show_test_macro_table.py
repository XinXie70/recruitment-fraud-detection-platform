"""Print Test macro comparison table (LR / Optimized BERT / FP-gate Ensemble).

VS Code / Cursor: open this file and click Run Python File.

Reads frozen metrics from sibling result folders (does not retrain).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

ROOT = CODE_DIR.parent
SPRINT3 = ROOT.parent
LR_METRICS = SPRINT3 / "LR" / "results" / "test_metrics.json"
ENSEMBLE_METRICS = ROOT / "results" / "test_metrics.json"
BERT_METRICS_CANDIDATES = (
    SPRINT3 / "BERT" / "results" / "metrics_test.json",
    SPRINT3
    / "BERT"
    / "results"
    / "bert-paper-protocol_test_metrics_bert_paper_protocol_maxlen512.json",
)


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bert_macro(path: Path) -> dict[str, float]:
    raw = load_json(path)
    if "metrics" in raw and isinstance(raw["metrics"], dict):
        metrics = raw["metrics"]
        report = metrics.get("classification_report") or {}
        macro = report.get("macro avg") or {}
        return {
            "macro_precision": float(
                raw.get("macro_precision", macro.get("precision", metrics.get("macro_precision", 0.0)))
            ),
            "macro_recall": float(
                raw.get("macro_recall", macro.get("recall", metrics.get("macro_recall", 0.0)))
            ),
            "macro_f1": float(raw.get("macro_f1", metrics.get("macro_f1", macro.get("f1-score", 0.0)))),
        }
    if "test_metrics" in raw:
        opt = raw["test_metrics"].get("threshold_optimized") or raw["test_metrics"]
        report = opt.get("classification_report") or {}
        macro = report.get("macro avg") or {}
        return {
            "macro_precision": float(macro.get("precision", 0.0)),
            "macro_recall": float(macro.get("recall", 0.0)),
            "macro_f1": float(opt.get("macro_f1", macro.get("f1-score", 0.0))),
        }
    raise ValueError(f"Unrecognized BERT metrics schema: {path}")


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    bert_path = first_existing(BERT_METRICS_CANDIDATES)
    missing = [p for p in (LR_METRICS, ENSEMBLE_METRICS) if not p.exists()]
    if bert_path is None:
        missing.append(Path("BERT/results/metrics_test.json"))
    if missing:
        print("Missing required metrics files:")
        for path in missing:
            print(f"  - {path}")
        print("Run the ensemble first: python run_fp_gate_ensemble.py")
        return 1

    lr = load_json(LR_METRICS)
    bert = load_bert_macro(bert_path)
    ens = load_json(ENSEMBLE_METRICS)

    rows = [
        ("LR", lr["macro_precision"], lr["macro_recall"], lr["macro_f1"]),
        (
            "Optimized BERT",
            bert["macro_precision"],
            bert["macro_recall"],
            bert["macro_f1"],
        ),
        (
            "FP-gate Ensemble",
            ens["macro_precision"],
            ens["macro_recall"],
            ens["macro_f1"],
        ),
    ]

    title = "Test 三方对比（Macro）"
    headers = ("Model", "Macro P", "Macro R", "Macro F1")
    col_w = [18, 10, 10, 10]

    print(title)
    print("=" * (sum(col_w) + 3))
    print(
        f"{headers[0]:<{col_w[0]}}"
        f"{headers[1]:>{col_w[1]}}"
        f"{headers[2]:>{col_w[2]}}"
        f"{headers[3]:>{col_w[3]}}"
    )
    print("-" * (sum(col_w) + 3))
    for name, p, r, f1 in rows:
        marker = " *" if name == "FP-gate Ensemble" else ""
        print(
            f"{name:<{col_w[0]}}"
            f"{fmt(p):>{col_w[1]}}"
            f"{fmt(r):>{col_w[2]}}"
            f"{fmt(f1):>{col_w[3]}}{marker}"
        )
    print("=" * (sum(col_w) + 3))
    print("* = FP-gate Ensemble (best across Macro P / R / F1 in current freeze)")
    print(f"Sources:\n  LR:       {LR_METRICS}\n  BERT:     {bert_path}\n  Ensemble: {ENSEMBLE_METRICS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
