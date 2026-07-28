"""Build and evaluate the three risk bands for Ensemble V1.

Validation mode selects the low/suspicious boundary using the agreed recall
rule. The suspicious/high boundary is read from the frozen ensemble config.

Test mode only applies the saved boundaries. It does not search for or change
either threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENSEMBLE_DIR = PROJECT_ROOT / "reports" / "models" / "ensemble_lr_bert"
REPORT_DIR = PROJECT_ROOT / "reports" / "models"

VALIDATION_INPUT = ENSEMBLE_DIR / "validation_predictions.csv"
TEST_INPUT = ENSEMBLE_DIR / "test_predictions.csv"
ENSEMBLE_CONFIG = ENSEMBLE_DIR / "ensemble_config.json"

RISK_CONFIG = REPORT_DIR / "risk_band_v1_config.json"
FEASIBILITY_OUTPUT = REPORT_DIR / "risk_band_v1_feasibility.csv"
VALIDATION_PREDICTIONS_OUTPUT = (
    REPORT_DIR / "risk_band_v1_validation_predictions.csv"
)
TEST_PREDICTIONS_OUTPUT = REPORT_DIR / "risk_band_v1_test_predictions.csv"
TEST_RESULTS_OUTPUT = REPORT_DIR / "risk_band_v1_test_results.csv"

EXPECTED_MODEL = "ensemble_lr_bert_class_weighted"
FEASIBILITY_TARGETS = (0.875, 0.90, 0.925, 0.95)


def load_predictions(path: Path) -> pd.DataFrame:
    """Load saved ensemble scores and run basic data checks."""
    required_columns = {"record_id", "model_name", "fraud_score", "true_label"}

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    data = pd.read_csv(path)
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing_columns)}"
        )

    if data["record_id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate record_id values")

    if not data["fraud_score"].between(0, 1).all():
        raise ValueError(f"{path.name} contains fraud scores outside [0, 1]")

    if not set(data["true_label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path.name} contains labels other than 0 and 1")

    model_names = set(data["model_name"].unique())
    if model_names != {EXPECTED_MODEL}:
        raise ValueError(
            f"Expected model_name '{EXPECTED_MODEL}', found {sorted(model_names)}"
        )

    return data


def load_high_threshold() -> float:
    """Read the frozen binary threshold from the ensemble configuration."""
    if not ENSEMBLE_CONFIG.exists():
        raise FileNotFoundError(f"Ensemble config not found: {ENSEMBLE_CONFIG}")

    with ENSEMBLE_CONFIG.open(encoding="utf-8") as file:
        config = json.load(file)

    threshold = float(config["selected_threshold"])
    if not 0 < threshold < 1:
        raise ValueError("The ensemble threshold must be between 0 and 1")
    return threshold


def non_low_fraud_recall(data: pd.DataFrame, low_threshold: float) -> float:
    """Return the share of fraudulent rows assigned outside the Low band."""
    fraud = data[data["true_label"] == 1]
    if fraud.empty:
        raise ValueError("Cannot calculate fraud recall because no fraud rows exist")
    return float((fraud["fraud_score"] >= low_threshold).mean())


def select_low_threshold(
    data: pd.DataFrame, minimum_recall: float, high_threshold: float
) -> float:
    """Select the highest score boundary that meets the minimum fraud recall."""
    candidates = sorted(
        data.loc[data["fraud_score"] < high_threshold, "fraud_score"].unique()
    )
    valid_candidates = [
        float(threshold)
        for threshold in candidates
        if non_low_fraud_recall(data, float(threshold)) >= minimum_recall
    ]

    if not valid_candidates:
        raise ValueError(
            f"No low threshold meets the recall target of {minimum_recall:.1%}"
        )
    return max(valid_candidates)


def add_risk_bands(
    data: pd.DataFrame, low_threshold: float, high_threshold: float
) -> pd.DataFrame:
    """Add risk score, risk level and frozen binary prediction columns."""
    output = data[
        ["record_id", "model_name", "fraud_score", "true_label"]
    ].copy()
    output["risk_score"] = output["fraud_score"] * 100
    output["risk_level"] = "Suspicious"
    output.loc[output["fraud_score"] < low_threshold, "risk_level"] = "Low"
    output.loc[output["fraud_score"] >= high_threshold, "risk_level"] = "High"
    output["binary_threshold"] = high_threshold
    output["binary_prediction"] = (
        output["fraud_score"] >= high_threshold
    ).astype(int)

    return output[
        [
            "record_id",
            "model_name",
            "fraud_score",
            "risk_score",
            "risk_level",
            "binary_threshold",
            "binary_prediction",
            "true_label",
        ]
    ]


def summarise_risk_bands(
    data: pd.DataFrame, low_threshold: float, high_threshold: float
) -> dict[str, float | int]:
    """Create one summary row for a pair of risk boundaries."""
    banded = add_risk_bands(data, low_threshold, high_threshold)
    total_rows = len(banded)

    summary: dict[str, float | int] = {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "non_low_fraud_recall": non_low_fraud_recall(data, low_threshold),
    }

    for risk_level, prefix in (
        ("Low", "low"),
        ("Suspicious", "suspicious"),
        ("High", "high"),
    ):
        band = banded[banded["risk_level"] == risk_level]
        fraud_count = int(band["true_label"].sum())
        summary[f"{prefix}_rows"] = len(band)
        summary[f"{prefix}_fraud"] = fraud_count
        summary[f"{prefix}_fraud_rate"] = (
            fraud_count / len(band) if len(band) else 0.0
        )
        summary[f"{prefix}_share"] = len(band) / total_rows

    summary["suspicious_review_rate"] = summary["suspicious_share"]
    return summary


def run_validation(minimum_recall: float) -> None:
    """Select and save the V1 risk boundaries using Validation only."""
    validation = load_predictions(VALIDATION_INPUT)
    high_threshold = load_high_threshold()

    selected_exact = select_low_threshold(
        validation, minimum_recall, high_threshold
    )
    selected_rounded = round(selected_exact, 4)
    rounded_recall = non_low_fraud_recall(validation, selected_rounded)
    if rounded_recall < minimum_recall:
        raise ValueError(
            "The rounded low threshold no longer meets the minimum recall rule"
        )

    feasibility_rows = []
    for target in FEASIBILITY_TARGETS:
        threshold = select_low_threshold(validation, target, high_threshold)
        if target == minimum_recall:
            threshold = selected_rounded
        feasibility_rows.append(
            summarise_risk_bands(validation, threshold, high_threshold)
        )

    feasibility = pd.DataFrame(feasibility_rows)
    feasibility_columns = [
        "low_threshold",
        "high_threshold",
        "non_low_fraud_recall",
        "low_rows",
        "low_fraud",
        "low_fraud_rate",
        "suspicious_rows",
        "suspicious_fraud",
        "suspicious_fraud_rate",
        "suspicious_review_rate",
        "high_rows",
        "high_fraud",
        "high_fraud_rate",
    ]
    feasibility[feasibility_columns].to_csv(FEASIBILITY_OUTPUT, index=False)

    risk_config = {
        "version": "v1",
        "ensemble_name": EXPECTED_MODEL,
        "score_scale": "0_to_1",
        "low_suspicious_threshold": selected_rounded,
        "suspicious_high_threshold": high_threshold,
        "selection_split": "validation",
        "low_threshold_rule": (
            "highest threshold with at least "
            f"{minimum_recall:.0%} non-low fraud recall"
        ),
        "minimum_non_low_fraud_recall": minimum_recall,
        "high_threshold_rule": "reuse frozen binary ensemble threshold",
        "test_used_for_selection": False,
    }
    with RISK_CONFIG.open("w", encoding="utf-8") as file:
        json.dump(risk_config, file, indent=2)
        file.write("\n")

    validation_output = add_risk_bands(
        validation, selected_rounded, high_threshold
    )
    validation_output.to_csv(VALIDATION_PREDICTIONS_OUTPUT, index=False)

    print(f"Saved risk config: {RISK_CONFIG}")
    print(f"Low/Suspicious threshold: {selected_rounded:.4f}")
    print(f"Suspicious/High threshold: {high_threshold:.2f}")
    print(f"Validation non-low fraud recall: {rounded_recall:.4f}")


def run_test() -> None:
    """Apply the frozen risk boundaries to Test without tuning them."""
    if not RISK_CONFIG.exists():
        raise FileNotFoundError(
            "Risk config not found. Run validation mode before test mode."
        )

    with RISK_CONFIG.open(encoding="utf-8") as file:
        config = json.load(file)

    if config.get("test_used_for_selection") is not False:
        raise ValueError("Risk config does not confirm that Test was held out")

    low_threshold = float(config["low_suspicious_threshold"])
    high_threshold = float(config["suspicious_high_threshold"])
    test = load_predictions(TEST_INPUT)

    test_output = add_risk_bands(test, low_threshold, high_threshold)
    test_output.to_csv(TEST_PREDICTIONS_OUTPUT, index=False)

    summary = summarise_risk_bands(test, low_threshold, high_threshold)
    results = []
    for risk_level, prefix in (
        ("Low", "low"),
        ("Suspicious", "suspicious"),
        ("High", "high"),
    ):
        rows = int(summary[f"{prefix}_rows"])
        fraud = int(summary[f"{prefix}_fraud"])
        results.append(
            {
                "risk_band": risk_level,
                "rows": rows,
                "fraud": fraud,
                "legitimate": rows - fraud,
                "fraud_rate": float(summary[f"{prefix}_fraud_rate"]),
                "share_of_test": float(summary[f"{prefix}_share"]),
            }
        )

    pd.DataFrame(results).to_csv(TEST_RESULTS_OUTPUT, index=False)

    print(f"Saved Test predictions: {TEST_PREDICTIONS_OUTPUT}")
    print(f"Saved Test summary: {TEST_RESULTS_OUTPUT}")
    print(
        "Test non-low fraud recall: "
        f"{float(summary['non_low_fraud_recall']):.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or evaluate the frozen Ensemble V1 risk bands."
    )
    parser.add_argument(
        "--mode",
        choices=("validation", "test"),
        required=True,
        help="Select thresholds on Validation or apply them to Test.",
    )
    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=0.90,
        help="Minimum Validation fraud recall outside Low (default: 0.90).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.minimum_recall <= 1:
        raise ValueError("--minimum-recall must be greater than 0 and at most 1")

    if args.mode == "validation":
        run_validation(args.minimum_recall)
    else:
        run_test()


if __name__ == "__main__":
    main()
