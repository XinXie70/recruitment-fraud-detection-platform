"""BERT primary + LR FP gate.

Default decision follows BERT with its decision threshold. When BERT predicts
fraud but the LR score is below an LR gate, flip the prediction to legitimate.

Searched on Validation only:
  - bert_threshold
  - lr_gate (flip BERT fraud -> legit if lr_score < lr_gate)
Selection: Fraud F1, then Fraud Recall, Fraud Precision, PR-AUC.
Test is evaluated once with frozen parameters.

Score used for PR-AUC / ROC-AUC reporting:
  keep BERT score, except gated flips are assigned min(bert_score, lr_gate)
  so ranking reflects the override (does not affect discrete F1 selection).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results"
CAPSTONE = ROOT.parent
LR_ROOT = CAPSTONE / "lr_none_paper_aligned_seed42"
BERT_ROOT = CAPSTONE / "retrain_paper_aligned_seed42_maxlen512"

LR_VAL = LR_ROOT / "results" / "validation_predictions.csv"
LR_TEST = LR_ROOT / "results" / "test_predictions.csv"
BERT_VAL = ROOT / "results" / "bert_validation_predictions.csv"
BERT_TEST = (
    BERT_ROOT / "results" / "predictions_bert_paper_protocol_maxlen512.csv"
)
LR_TEST_METRICS = LR_ROOT / "results" / "test_metrics.json"
BERT_TEST_METRICS = (
    BERT_ROOT
    / "results"
    / "bert-paper-protocol_test_metrics_bert_paper_protocol_maxlen512.json"
)

BERT_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
LR_GATES = np.unique(
    np.concatenate(
        [
            np.array([0.0]),  # never gate (= pure BERT)
            np.round(np.arange(0.01, 0.51, 0.01), 2),
            np.round(np.arange(0.55, 1.01, 0.05), 2),
        ]
    )
)


def align_scores(lr_path: Path, bert_path: Path, split: str) -> pd.DataFrame:
    lr = pd.read_csv(lr_path).rename(columns={"fraud_score": "lr_score"})
    bert = pd.read_csv(bert_path).rename(
        columns={
            "original_id": "record_id",
            "true_label": "label",
            "fraud_probability": "bert_score",
            "fraud_score": "bert_score",
        }
    )
    lr = lr[["record_id", "label", "lr_score"]]
    bert = bert[["record_id", "label", "bert_score"]]
    aligned = lr.merge(bert, on=["record_id", "label"], how="inner", validate="one_to_one")
    if len(aligned) != len(lr) or len(aligned) != len(bert):
        raise ValueError(f"{split}: LR and BERT rows are not fully aligned")
    if aligned["record_id"].duplicated().any():
        raise ValueError(f"{split}: duplicate record IDs found")
    return aligned


def apply_gate(
    lr_scores: np.ndarray,
    bert_scores: np.ndarray,
    bert_threshold: float,
    lr_gate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bert_pred = (bert_scores >= bert_threshold).astype(int)
    gated = (bert_pred == 1) & (lr_scores < lr_gate)
    final_pred = bert_pred.copy()
    final_pred[gated] = 0
    # Ranking score: suppress gated fraud scores for AUC reporting.
    ranking_scores = bert_scores.copy()
    ranking_scores[gated] = np.minimum(bert_scores[gated], lr_scores[gated])
    return final_pred, gated, ranking_scores


def metrics_from_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    ranking_scores: np.ndarray,
) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, ranking_scores)),
        "pr_auc": float(average_precision_score(labels, ranking_scores)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "fraud_precision": float(precision[1]),
        "fraud_recall": float(recall[1]),
        "fraud_f1": float(f1[1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    validation = align_scores(LR_VAL, BERT_VAL, "Validation")
    test = align_scores(LR_TEST, BERT_TEST, "Test")
    if set(validation["record_id"]) & set(test["record_id"]):
        raise ValueError("Validation and Test IDs overlap")

    y_val = validation["label"].to_numpy(dtype=int)
    lr_val = validation["lr_score"].to_numpy()
    bert_val = validation["bert_score"].to_numpy()

    sweep_rows = []
    for bert_threshold in BERT_THRESHOLDS:
        for lr_gate in LR_GATES:
            preds, gated, ranks = apply_gate(lr_val, bert_val, float(bert_threshold), float(lr_gate))
            metrics = metrics_from_predictions(y_val, preds, ranks)
            sweep_rows.append(
                {
                    "bert_threshold": float(bert_threshold),
                    "lr_gate": float(lr_gate),
                    "gate_flips": int(gated.sum()),
                    "gate_rate": float(gated.mean()),
                    **metrics,
                }
            )

    sweep = pd.DataFrame(sweep_rows)
    selected = sweep.sort_values(
        ["fraud_f1", "fraud_recall", "fraud_precision", "pr_auc"],
        ascending=False,
    ).iloc[0]

    bert_threshold = float(selected["bert_threshold"])
    lr_gate = float(selected["lr_gate"])

    val_preds, val_gated, val_ranks = apply_gate(lr_val, bert_val, bert_threshold, lr_gate)
    validation_metrics = metrics_from_predictions(y_val, val_preds, val_ranks)
    validation_metrics["bert_threshold"] = bert_threshold
    validation_metrics["lr_gate"] = lr_gate
    validation_metrics["gate_flips"] = int(val_gated.sum())

    # Pure BERT best on same threshold grid (lr_gate=0 => no flips).
    bert_only = sweep[np.isclose(sweep["lr_gate"], 0.0)].sort_values(
        ["fraud_f1", "fraud_recall", "fraud_precision", "pr_auc"],
        ascending=False,
    ).iloc[0]

    y_test = test["label"].to_numpy(dtype=int)
    lr_test = test["lr_score"].to_numpy()
    bert_test = test["bert_score"].to_numpy()
    test_preds, test_gated, test_ranks = apply_gate(
        lr_test, bert_test, bert_threshold, lr_gate
    )
    test_metrics = metrics_from_predictions(y_test, test_preds, test_ranks)
    test_metrics["bert_threshold"] = bert_threshold
    test_metrics["lr_gate"] = lr_gate
    test_metrics["gate_flips"] = int(test_gated.sum())

    # Also report pure-BERT at the selected bert_threshold for apples-to-apples.
    bert_at_selected_preds = (bert_test >= bert_threshold).astype(int)
    bert_at_selected = metrics_from_predictions(y_test, bert_at_selected_preds, bert_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUTPUT_DIR / "validation_sweep.csv", index=False)
    validation.assign(
        prediction=val_preds,
        gated_flip=val_gated.astype(int),
        ranking_score=val_ranks,
    ).to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    test.assign(
        prediction=test_preds,
        gated_flip=test_gated.astype(int),
        ranking_score=test_ranks,
    ).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    config = {
        "experiment": "ensemble_bert_fp_gate_lr_none_maxlen512",
        "method": "BERT decision + flip to legit when LR_score < lr_gate",
        "bert_threshold": bert_threshold,
        "lr_gate": lr_gate,
        "selection_set": "validation",
        "selection_metric": "fraud_f1",
        "test_used_for_selection": False,
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "validation_gate_flips": int(val_gated.sum()),
        "test_gate_flips": int(test_gated.sum()),
        "validation_bert_only_best_fraud_f1": float(bert_only["fraud_f1"]),
        "validation_bert_only_best_threshold": float(bert_only["bert_threshold"]),
        "validation_selected_fraud_f1": float(validation_metrics["fraud_f1"]),
    }
    for filename, payload in (
        ("config.json", config),
        ("validation_metrics.json", validation_metrics),
        ("test_metrics.json", test_metrics),
        ("test_bert_at_selected_threshold.json", bert_at_selected),
    ):
        (OUTPUT_DIR / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lr_ref = json.loads(LR_TEST_METRICS.read_text(encoding="utf-8"))
    bert_raw = json.loads(BERT_TEST_METRICS.read_text(encoding="utf-8"))
    bert_opt = bert_raw["test_metrics"]["threshold_optimized"]

    improved = test_metrics["fraud_f1"] > bert_opt["fraud_f1"] + 1e-12
    results_md = f"""# BERT + LR FP-gate ensemble

## Method

1. Predict with BERT using `bert_threshold`
2. If BERT predicts fraud **and** `LR_score < lr_gate`, flip to legitimate
3. Select `(bert_threshold, lr_gate)` on Validation by Fraud F1
4. Freeze and evaluate once on Test

## Selected configuration

- BERT threshold: `{bert_threshold:.2f}`
- LR gate: `{lr_gate:.2f}`
- Validation gate flips: `{int(val_gated.sum())}/{len(validation)}`
- Test gate flips: `{int(test_gated.sum())}/{len(test)}`
- Validation Fraud F1: `{validation_metrics["fraud_f1"]:.4f}`
- Validation pure-BERT best Fraud F1: `{float(bert_only["fraud_f1"]):.4f}` (thr {float(bert_only["bert_threshold"]):.2f})

## Test comparison

| Model | Fraud P | Fraud R | Fraud F1 | Macro P | Macro R | Macro F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Improved LR (`None`) | {lr_ref["fraud_precision"]:.4f} | {lr_ref["fraud_recall"]:.4f} | {lr_ref["fraud_f1"]:.4f} | {lr_ref["macro_precision"]:.4f} | {lr_ref["macro_recall"]:.4f} | {lr_ref["macro_f1"]:.4f} | {lr_ref["pr_auc"]:.4f} | {lr_ref["roc_auc"]:.4f} |
| BERT max_length=512 (paper thr) | {bert_opt["fraud_precision"]:.4f} | {bert_opt["fraud_recall"]:.4f} | {bert_opt["fraud_f1"]:.4f} | {bert_opt["classification_report"]["macro avg"]["precision"]:.4f} | {bert_opt["classification_report"]["macro avg"]["recall"]:.4f} | {bert_opt["macro_f1"]:.4f} | {bert_opt["pr_auc"]:.4f} | {bert_opt["roc_auc"]:.4f} |
| BERT at selected thr `{bert_threshold:.2f}` (no gate) | {bert_at_selected["fraud_precision"]:.4f} | {bert_at_selected["fraud_recall"]:.4f} | {bert_at_selected["fraud_f1"]:.4f} | {bert_at_selected["macro_precision"]:.4f} | {bert_at_selected["macro_recall"]:.4f} | {bert_at_selected["macro_f1"]:.4f} | {bert_at_selected["pr_auc"]:.4f} | {bert_at_selected["roc_auc"]:.4f} |
| BERT + LR FP-gate | {test_metrics["fraud_precision"]:.4f} | {test_metrics["fraud_recall"]:.4f} | {test_metrics["fraud_f1"]:.4f} | {test_metrics["macro_precision"]:.4f} | {test_metrics["macro_recall"]:.4f} | {test_metrics["macro_f1"]:.4f} | {test_metrics["pr_auc"]:.4f} | {test_metrics["roc_auc"]:.4f} |

Fraud F1 vs paper BERT: {"higher" if improved else "not higher"}.

Confusion matrix on Test: TN {test_metrics["tn"]}, FP {test_metrics["fp"]}, FN {test_metrics["fn"]}, TP {test_metrics["tp"]}.
"""
    (OUTPUT_DIR / "RESULTS.md").write_text(results_md, encoding="utf-8")
    (ROOT / "README.md").write_text(
        """# BERT + LR FP-gate

BERT 为主；当 BERT 判欺诈且 LR 分数低于门控阈值时，改判正常（压 FP）。

对外风险输出口径见：[RISK_SCORE_AND_LEVEL.md](./RISK_SCORE_AND_LEVEL.md)

- **Risk score（0–1）** = `bert_score`（不改写）
- **Risk level** = 无 / 低 / 高（分数阈值 + FP-gate 降档）

```powershell
. E:\\ml\\activate.ps1
cd F:\\better-BERT\\capstone-project-26t2-9900-h09c-almond\\ensemble_bert_fp_gate_lr_none_maxlen512\\code
python run_fp_gate_ensemble.py
```
""",
        encoding="utf-8",
    )

    print(json.dumps({"config": config, "validation": validation_metrics, "test": test_metrics}, indent=2))
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
