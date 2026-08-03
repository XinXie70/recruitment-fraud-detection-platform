"""Select and apply the three-level risk boundaries.

Selection uses Validation only. BERT is the primary risk-scoring model. LR is
used only as the existing false-positive gate for BERT High candidates. Test
mode applies the frozen configuration and never selects thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
VALIDATION_PREDICTIONS = RESULTS / "validation_predictions.csv"
TEST_PREDICTIONS = RESULTS / "test_predictions.csv"
ENSEMBLE_CONFIG = RESULTS / "config.json"
HIGH_SWEEP = RESULTS / "validation_sweep.csv"
BOUNDARY_CONFIG = RESULTS / "risk_boundary_config.json"

MINIMUM_NON_LOW_FRAUD_RECALL = 0.90

# The BERT validation scores are concentrated near zero, so the grid is finer
# in that region. The grid is fixed before Test evaluation.
BERT_LOW_THRESHOLDS = np.unique(
    np.concatenate(
        [
            np.round(np.arange(0.00001, 0.00101, 0.00001), 5),
            np.round(np.arange(0.0011, 0.0051, 0.0001), 4),
            np.round(np.arange(0.006, 0.051, 0.001), 3),
            np.round(np.arange(0.06, 0.30, 0.01), 2),
        ]
    )
)


def load_predictions(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"record_id", "label", "bert_score", "lr_score"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if data["record_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate record IDs")
    if not data["label"].isin([0, 1]).all():
        raise ValueError(f"{path} contains labels other than 0 and 1")
    return data


def load_high_rule() -> tuple[float, float]:
    config = json.loads(ENSEMBLE_CONFIG.read_text(encoding="utf-8"))
    return float(config["bert_threshold"]), float(config["lr_gate"])


def verify_high_rule(bert_threshold: float, lr_gate: float) -> dict:
    """Confirm that the frozen High rule is the selected Validation row."""
    sweep = pd.read_csv(HIGH_SWEEP)
    best = sweep.sort_values(
        [
            "fraud_f1",
            "fraud_recall",
            "fraud_precision",
            "pr_auc",
            "bert_threshold",
            "lr_gate",
        ],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]
    if not np.isclose(float(best["bert_threshold"]), bert_threshold):
        raise ValueError("Frozen BERT threshold is not the selected Validation value")
    if not np.isclose(float(best["lr_gate"]), lr_gate):
        raise ValueError("Frozen LR gate is not the selected Validation value")
    return {
        "bert_threshold": bert_threshold,
        "lr_gate": lr_gate,
        "validation_fraud_precision": float(best["fraud_precision"]),
        "validation_fraud_recall": float(best["fraud_recall"]),
        "validation_fraud_f1": float(best["fraud_f1"]),
    }


def assign_levels(
    data: pd.DataFrame,
    bert_high_threshold: float,
    lr_gate: float,
    bert_low_threshold: float,
) -> pd.DataFrame:
    if not 0 < bert_low_threshold < bert_high_threshold < 1:
        raise ValueError("Expected 0 < Low threshold < High threshold < 1")

    output = data.copy()
    high = (output["bert_score"] >= bert_high_threshold) & (
        output["lr_score"] >= lr_gate
    )
    low = output["bert_score"] < bert_low_threshold
    if (high & low).any():
        raise ValueError("Low and High rules overlap")

    output["risk_score"] = output["bert_score"]
    output["risk_level"] = "Suspicious"
    output.loc[low, "risk_level"] = "Low"
    output.loc[high, "risk_level"] = "High"
    output["high_rule_met"] = high.astype(int)
    output["low_rule_met"] = low.astype(int)
    return output


def summarise_levels(output: pd.DataFrame) -> dict:
    total_fraud = int((output["label"] == 1).sum())
    summary = {
        "rows": int(len(output)),
        "fraudulent": total_fraud,
        "legitimate": int((output["label"] == 0).sum()),
    }
    for level in ["Low", "Suspicious", "High"]:
        selected = output[output["risk_level"] == level]
        summary[f"{level.lower()}_total"] = int(len(selected))
        summary[f"{level.lower()}_fraud"] = int((selected["label"] == 1).sum())
        summary[f"{level.lower()}_legitimate"] = int(
            (selected["label"] == 0).sum()
        )
    summary["non_low_fraud"] = total_fraud - summary["low_fraud"]
    summary["non_low_fraud_recall"] = (
        summary["non_low_fraud"] / total_fraud if total_fraud else 0.0
    )
    if (
        summary["low_total"]
        + summary["suspicious_total"]
        + summary["high_total"]
        != summary["rows"]
    ):
        raise ValueError("Risk-level counts do not cover every row exactly once")
    return summary


def build_low_tradeoff(
    validation: pd.DataFrame, bert_high_threshold: float, lr_gate: float
) -> pd.DataFrame:
    rows = []
    for bert_low_threshold in BERT_LOW_THRESHOLDS:
        if bert_low_threshold >= bert_high_threshold:
            continue
        output = assign_levels(
            validation,
            bert_high_threshold,
            lr_gate,
            float(bert_low_threshold),
        )
        summary = summarise_levels(output)
        rows.append(
            {
                "bert_low_threshold": float(bert_low_threshold),
                "meets_90pct_target": (
                    summary["non_low_fraud_recall"]
                    >= MINIMUM_NON_LOW_FRAUD_RECALL
                ),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def select_low_boundary(tradeoff: pd.DataFrame, minimum_recall: float) -> pd.Series:
    eligible = tradeoff[tradeoff["non_low_fraud_recall"] >= minimum_recall]
    if eligible.empty:
        raise ValueError("No BERT Low boundary meets the fraud recall target")

    # In a one-dimensional score rule, increasing the Low threshold moves more
    # advertisements into Low. Select the highest threshold that still meets
    # the pre-set fraud safety target.
    return eligible.sort_values("bert_low_threshold", ascending=False).iloc[0]


def target_comparison(tradeoff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in [0.90, 0.925, 0.95, 0.975]:
        selected = select_low_boundary(tradeoff, target)
        rows.append({"minimum_recall_target": target, **selected.to_dict()})
    return pd.DataFrame(rows)


def write_report(high: dict, selected_low: dict, comparison: pd.DataFrame) -> None:
    comparison_rows = []
    for _, row in comparison.iterrows():
        comparison_rows.append(
            "| "
            f"{row['minimum_recall_target']:.1%} | "
            f"{row['bert_low_threshold']:.5f} | "
            f"{int(row['non_low_fraud'])}/{int(row['fraudulent'])} "
            f"({row['non_low_fraud_recall']:.2%}) | "
            f"{int(row['low_fraud'])} | "
            f"{int(row['suspicious_total'])} | "
            f"{int(row['suspicious_legitimate'])} |"
        )

    report = f"""# Three-Level Risk Boundary Report

## Selection protocol

- Selection data: Validation only
- Test used for selection: No
- Labels: 0 = legitimate, 1 = fraudulent
- Suspicious is an operational review band, not a ground-truth class
- BERT remains the primary risk-scoring model
- LR remains only the existing false-positive gate for High candidates

## High rule

The existing FP-gate search selected the pair with the highest Validation
Fraud F1:

```text
High if BERT score >= {high['bert_threshold']:.2f} and LR score >= {high['lr_gate']:.2f}
```

Validation Fraud Precision: {high['validation_fraud_precision']:.4f}  
Validation Fraud Recall: {high['validation_fraud_recall']:.4f}  
Validation Fraud F1: {high['validation_fraud_f1']:.4f}

## Low boundary

The Low boundary uses only the official BERT risk score. This preserves the
frozen ensemble design: LR is not introduced as a second Low-risk decision
mechanism.

Candidate BERT thresholds were compared on Validation. The selected boundary
is the highest threshold that keeps at least 90% of known fraud
advertisements outside Low.

```text
Low if BERT score < {selected_low['bert_low_threshold']:.4f}
```

Selected Validation result:

- Fraud kept outside Low: {int(selected_low['non_low_fraud'])}/{int(selected_low['fraudulent'])} ({selected_low['non_low_fraud_recall']:.2%})
- Fraud left in Low: {int(selected_low['low_fraud'])}
- Suspicious advertisements: {int(selected_low['suspicious_total'])}
- Legitimate advertisements in Suspicious: {int(selected_low['suspicious_legitimate'])}

## Validation trade-off

| Minimum target | BERT Low boundary | Fraud kept out of Low | Fraud in Low | Suspicious | Legitimate in Suspicious |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

## Final three-level rule

```text
if BERT score >= {high['bert_threshold']:.2f} and LR score >= {high['lr_gate']:.2f}:
    High
elif BERT score < {selected_low['bert_low_threshold']:.4f}:
    Low
else:
    Suspicious
```

The thresholds must be frozen before Test evaluation. The model scores are
operational scores and should not be interpreted as calibrated probabilities.
"""
    (RESULTS / "RISK_BOUNDARY_REPORT.md").write_text(report, encoding="utf-8")


def select_on_validation() -> None:
    validation = load_predictions(VALIDATION_PREDICTIONS)
    bert_high_threshold, lr_gate = load_high_rule()
    high = verify_high_rule(bert_high_threshold, lr_gate)

    tradeoff = build_low_tradeoff(validation, bert_high_threshold, lr_gate)
    selected = select_low_boundary(tradeoff, MINIMUM_NON_LOW_FRAUD_RECALL)
    selected_low = selected.to_dict()
    comparison = target_comparison(tradeoff)

    selected_output = assign_levels(
        validation,
        bert_high_threshold,
        lr_gate,
        float(selected["bert_low_threshold"]),
    )
    selected_summary = summarise_levels(selected_output)

    config = {
        "selection_set": "validation",
        "test_used_for_selection": False,
        "minimum_non_low_fraud_recall": MINIMUM_NON_LOW_FRAUD_RECALL,
        "risk_score": "bert_score",
        "ensemble_roles": {
            "bert": "primary risk-scoring model",
            "lr": "false-positive gate for BERT High candidates only",
        },
        "high_rule": {
            "bert_threshold": bert_high_threshold,
            "lr_gate": lr_gate,
            "selection_metric": "maximum validation fraud_f1",
        },
        "low_rule": {
            "method": "BERT primary risk score only",
            "bert_threshold": float(selected["bert_low_threshold"]),
            "selection_rule": (
                "highest BERT threshold with minimum non-low fraud recall"
            ),
        },
        "validation_summary": selected_summary,
    }

    tradeoff.to_csv(RESULTS / "low_boundary_tradeoff.csv", index=False)
    comparison.to_csv(RESULTS / "low_boundary_target_comparison.csv", index=False)
    selected_output.to_csv(RESULTS / "validation_risk_levels.csv", index=False)
    BOUNDARY_CONFIG.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(high, selected_low, comparison)

    print("Boundary selection completed using Validation only.")
    print(
        f"High: BERT >= {bert_high_threshold:.2f} and LR >= {lr_gate:.2f}"
    )
    print(f"Low: BERT < {selected['bert_low_threshold']:.4f}")
    print(
        "Non-Low Fraud Recall: "
        f"{selected_summary['non_low_fraud_recall']:.2%}"
    )


def apply_to_test() -> None:
    if not BOUNDARY_CONFIG.exists():
        raise FileNotFoundError("Run --mode select before applying to Test")
    config = json.loads(BOUNDARY_CONFIG.read_text(encoding="utf-8"))
    test = load_predictions(TEST_PREDICTIONS)
    output = assign_levels(
        test,
        float(config["high_rule"]["bert_threshold"]),
        float(config["high_rule"]["lr_gate"]),
        float(config["low_rule"]["bert_threshold"]),
    )
    output.to_csv(RESULTS / "test_risk_levels.csv", index=False)
    summary = summarise_levels(output)
    (RESULTS / "test_risk_level_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Frozen boundaries applied to Test. No threshold search was run.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["select", "apply-test"], default="select"
    )
    args = parser.parse_args()
    if args.mode == "select":
        select_on_validation()
    else:
        apply_to_test()


if __name__ == "__main__":
    main()
