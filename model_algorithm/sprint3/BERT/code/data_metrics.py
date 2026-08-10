"""Text preprocessing, datasets, metrics, and evaluation plots."""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import Dataset

from config import (
    FIELD_TAG_MAP,
    GROUP_COLUMN,
    ID_COLUMN,
    LABEL_COLUMN,
    PREFER_COMBINED_TEXT,
    TAGGED_FIELD_ORDER,
    TEST_CSV,
    TEXT_COLUMN,
    TEXT_FIELDS,
    TRAIN_CSV,
    USE_TAGGED_FORMAT,
    VALIDATION_CSV,
)

# ========================================================================
# PREPROCESSING
# ========================================================================

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n\s*\n+")
_URL = re.compile(
    r"(https?://\S+|www\.\S+)",
    re.IGNORECASE,
)
_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}"
)


def missing_to_empty(value: Any) -> str:
    """Convert missing / literal 'nan' values to empty string."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def clean_text(
    value: Any,
    *,
    replace_url: bool = True,
    replace_email: bool = True,
    replace_phone: bool = False,
) -> str:
    """Light cleaning suitable for BERT (no stemming / stopword removal)."""
    text = missing_to_empty(value)
    if not text:
        return ""

    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub(" ", text)

    if replace_url:
        text = _URL.sub(" [URL] ", text)
    if replace_email:
        text = _EMAIL.sub(" [EMAIL] ", text)
    if replace_phone:
        text = _PHONE.sub(" [PHONE] ", text)

    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n", text)
    return text.strip()


def build_tagged_text(row: Dict[str, Any], field_order: Optional[Iterable[str]] = None) -> str:
    """Concatenate available fields with fixed tags, never dropping the row."""
    order = list(field_order) if field_order is not None else list(TAGGED_FIELD_ORDER)
    parts: List[str] = []
    for field in order:
        if field not in row:
            continue
        cleaned = clean_text(row.get(field, ""))
        if not cleaned:
            continue
        tag = FIELD_TAG_MAP.get(field, field.upper())
        parts.append(f"[{tag}] {cleaned}")
    return "\n".join(parts)


def build_tagged_from_combined(combined: str) -> str:
    """Best-effort section tags when only `combined_text` is available.

    Pipeline text is newline-joined from TEXT_FIELDS with empty fields dropped.
    Exact field recovery is impossible when fields contain newlines, so we map:
      - line 0 -> TITLE
      - line 1 -> COMPANY PROFILE (if present)
      - remaining lines split across DESCRIPTION / REQUIREMENTS / BENEFITS
    """
    text = clean_text(combined, replace_url=True, replace_email=True)
    if not text:
        return ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return f"[TITLE] {lines[0]}"

    parts: List[str] = [f"[TITLE] {lines[0]}"]
    rest = lines[1:]
    if len(rest) == 1:
        parts.append(f"[DESCRIPTION] {rest[0]}")
        return "\n".join(parts)

    parts.append(f"[COMPANY PROFILE] {rest[0]}")
    body = rest[1:]
    if not body:
        return "\n".join(parts)
    if len(body) == 1:
        parts.append(f"[DESCRIPTION] {body[0]}")
        return "\n".join(parts)

    # Split remaining lines into up to three body sections.
    n = len(body)
    cuts = [max(1, n // 3), max(2, (2 * n) // 3)]
    desc = "\n".join(body[: cuts[0]])
    reqs = "\n".join(body[cuts[0] : cuts[1]])
    bens = "\n".join(body[cuts[1] :])
    if desc:
        parts.append(f"[DESCRIPTION] {desc}")
    if reqs:
        parts.append(f"[REQUIREMENTS] {reqs}")
    if bens:
        parts.append(f"[BENEFITS] {bens}")
    return "\n".join(parts)


def build_plain_combined(row: Dict[str, Any], fields: Optional[Iterable[str]] = None) -> str:
    """Newline-join cleaned non-empty fields (matches shared data contract)."""
    order = list(fields) if fields is not None else list(TEXT_FIELDS)
    parts = [clean_text(row.get(field, "")) for field in order]
    return "\n".join(p for p in parts if p)


def extract_text_from_row(
    row: Dict[str, Any],
    *,
    prefer_combined: bool = PREFER_COMBINED_TEXT,
    use_tagged: bool = USE_TAGGED_FORMAT,
) -> str:
    """Resolve model input text for one advertisement row."""
    available = [f for f in TAGGED_FIELD_ORDER if f in row and missing_to_empty(row.get(f, ""))]

    # Tagged format takes priority when enabled.
    if use_tagged:
        if available:
            return build_tagged_text(row)
        if TEXT_COLUMN in row:
            tagged = build_tagged_from_combined(row.get(TEXT_COLUMN, ""))
            if tagged:
                return tagged

    if prefer_combined and TEXT_COLUMN in row:
        combined = clean_text(row.get(TEXT_COLUMN, ""), replace_url=True, replace_email=True)
        if combined:
            return combined

    if available:
        return build_plain_combined(row, available)

    if TEXT_COLUMN in row:
        combined = clean_text(row.get(TEXT_COLUMN, ""), replace_url=True, replace_email=True)
        if combined:
            return combined

    # Fallback: any string-like leftover columns except ids/labels.
    skip = {
        "label",
        "fraudulent",
        "record_id",
        "group_id",
        "in_balanced_dataset",
        "model_text",
        "split",
        "row_index",
    }
    leftovers = [
        clean_text(v)
        for k, v in row.items()
        if k not in skip and isinstance(v, (str, int, float))
    ]
    return "\n".join(x for x in leftovers if x)


def prepare_dataframe_text(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a standardised `model_text` column."""
    out = df.copy()
    texts = [
        extract_text_from_row(row._asdict() if hasattr(row, "_asdict") else row.to_dict())
        for _, row in out.iterrows()
    ]
    out["model_text"] = texts
    return out


def derive_title_preview(text: str, max_chars: int = 120) -> str:
    """Best-effort title from first line of combined text."""
    if not text:
        return ""
    first = text.split("\n", 1)[0].strip()
    return first[:max_chars]


def derive_company_preview(text: str, max_chars: int = 120) -> str:
    """Best-effort company snippet from second non-empty line."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return ""
    return lines[1][:max_chars]

# ========================================================================
# METRICS
# ========================================================================

def binary_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute a full metric suite focused on the fraud class (label=1)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    metrics: Dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "legitimate_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "legitimate_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "legitimate_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "fraud_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate", "Fraudulent"],
            zero_division=0,
            output_dict=True,
        ),
    }

    # ROC / PR need both classes present in y_true for a meaningful score.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def threshold_sweep(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
    *,
    step: float = 0.01,
) -> pd.DataFrame:
    """Evaluate fraud metrics across candidate thresholds (default step 0.01)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, float(step)), 4)
        # Always include the endpoints used historically for comparability.
        thresholds = sorted({0.01, *thresholds.tolist(), 0.99})
    rows = []
    for thr in thresholds:
        m = binary_metrics(y_true, y_prob, threshold=float(thr))
        rows.append(
            {
                "threshold": float(thr),
                "fraud_precision": m["fraud_precision"],
                "fraud_recall": m["fraud_recall"],
                "fraud_f1": m["fraud_f1"],
                "false_positive": m["confusion_matrix"]["fp"],
                "false_negative": m["confusion_matrix"]["fn"],
                "tn": m["confusion_matrix"]["tn"],
                "tp": m["confusion_matrix"]["tp"],
                "macro_f1": m["macro_f1"],
                "accuracy": m["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    mode: str = "max_fraud_f1",
    min_fraud_recall: Optional[float] = None,
    step: float = 0.01,
    f_beta: float = 0.5,
    f1_tolerance: float = 0.01,
) -> Tuple[float, pd.DataFrame]:
    """Choose a classification threshold on the VALIDATION set only.

    Modes:
      - max_fraud_f1: threshold with highest fraud F1
        (ties broken by higher precision, then higher threshold)
      - max_fbeta: maximize fraud F-beta (beta<1 favors precision / higher thr)
      - near_max_f1_prefer_precision: among thresholds within f1_tolerance of
        the best fraud F1, pick highest fraud precision then highest threshold
      - min_recall_then_f1: among thresholds with fraud_recall >= min_fraud_recall,
        pick highest fraud F1 (falls back to unconstrained max F1 if none qualify)
      - min_recall_then_precision: among thresholds meeting min fraud recall,
        pick highest fraud precision (falls back to max fraud F1 if none qualify)
    """
    sweep = threshold_sweep(y_true, y_prob, step=step)

    if mode == "min_recall_then_f1" and min_fraud_recall is not None:
        eligible = sweep[sweep["fraud_recall"] >= float(min_fraud_recall)]
        if len(eligible) > 0:
            best = eligible.sort_values(
                ["fraud_f1", "fraud_precision", "threshold"],
                ascending=[False, False, False],
            ).iloc[0]
            return float(best["threshold"]), sweep

    if mode == "min_recall_then_precision" and min_fraud_recall is not None:
        eligible = sweep[sweep["fraud_recall"] >= float(min_fraud_recall)]
        if len(eligible) > 0:
            best = eligible.sort_values(
                ["fraud_precision", "fraud_f1", "threshold"],
                ascending=[False, False, False],
            ).iloc[0]
            return float(best["threshold"]), sweep

    if mode == "max_fbeta":
        beta = float(f_beta)
        beta2 = beta * beta
        scored = sweep.copy()
        denom = beta2 * scored["fraud_precision"] + scored["fraud_recall"]
        scored["fraud_fbeta"] = np.where(
            denom > 0,
            (1.0 + beta2) * scored["fraud_precision"] * scored["fraud_recall"] / denom,
            0.0,
        )
        best = scored.sort_values(
            ["fraud_fbeta", "fraud_precision", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        return float(best["threshold"]), sweep

    if mode == "near_max_f1_prefer_precision":
        max_f1 = float(sweep["fraud_f1"].max())
        eligible = sweep[sweep["fraud_f1"] >= max_f1 - float(f1_tolerance)]
        best = eligible.sort_values(
            ["fraud_precision", "threshold", "fraud_f1"],
            ascending=[False, False, False],
        ).iloc[0]
        return float(best["threshold"]), sweep

    # Default: max fraud F1; prefer precision / higher threshold on ties.
    best = sweep.sort_values(
        ["fraud_f1", "fraud_precision", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best["threshold"]), sweep


def build_predictions_frame(
    *,
    record_ids: Sequence[str],
    titles: Sequence[str],
    companies: Sequence[str],
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float,
    model_name: str,
    imbalance_strategy: str,
    row_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    if row_indices is None:
        row_indices = list(range(n))
    return pd.DataFrame(
        {
            "sample_index": list(row_indices),
            "original_id": list(record_ids),
            "company": list(companies),
            "title": list(titles),
            "true_label": y_true,
            "predicted_label": y_pred,
            "legitimate_probability": 1.0 - y_prob,
            "fraud_probability": y_prob,
            "threshold": threshold,
            "correct": (y_true == y_pred).astype(int),
            "model_name": model_name,
            "imbalance_strategy": imbalance_strategy,
        }
    )


def build_error_analysis(
    preds: pd.DataFrame,
    texts: Sequence[str],
    *,
    high_conf: float = 0.85,
    near_margin: float = 0.05,
) -> pd.DataFrame:
    """Flag FP / FN / high-confidence errors / near-threshold samples."""
    df = preds.copy()
    df["text_preview"] = [str(t)[:300].replace("\n", " ") for t in texts]
    thr = float(df["threshold"].iloc[0]) if len(df) else 0.5

    def error_type(row) -> str:
        if row["true_label"] == 0 and row["predicted_label"] == 1:
            base = "false_positive"
        elif row["true_label"] == 1 and row["predicted_label"] == 0:
            base = "false_negative"
        else:
            base = "correct"
        extras = []
        if base != "correct" and (
            (row["predicted_label"] == 1 and row["fraud_probability"] >= high_conf)
            or (row["predicted_label"] == 0 and row["fraud_probability"] <= 1 - high_conf)
        ):
            extras.append("high_confidence_error")
        if abs(row["fraud_probability"] - thr) <= near_margin:
            extras.append("near_threshold")
        if extras:
            return base + "|" + "|".join(extras)
        return base

    df["error_type"] = df.apply(error_type, axis=1)
    df = df.rename(
        columns={
            "original_id": "sample_id",
            "true_label": "original_label",
        }
    )
    cols = [
        "sample_id",
        "title",
        "company",
        "original_label",
        "predicted_label",
        "fraud_probability",
        "threshold",
        "error_type",
        "text_preview",
    ]
    return df[cols]


def save_comparison_row(
    path: Path,
    row: Dict[str, Any],
) -> pd.DataFrame:
    """Append / upsert one model row into model_comparison.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "Model",
        "Imbalance method",
        "Accuracy",
        "Fraud Precision",
        "Fraud Recall",
        "Fraud F1",
        "Macro F1",
        "ROC-AUC",
        "PR-AUC",
        "Inference time (s)",
        "Notes",
    ]
    if path.exists():
        table = pd.read_csv(path)
    else:
        table = pd.DataFrame(columns=cols)

    key_model = row.get("Model")
    key_imb = row.get("Imbalance method")
    mask = (table["Model"] == key_model) & (table["Imbalance method"] == key_imb)
    for c in cols:
        if c not in table.columns:
            table[c] = np.nan
    if mask.any():
        for k, v in row.items():
            table.loc[mask, k] = v
    else:
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)
    return table


def plot_training_curves(history: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "train_loss" in history.columns:
        ax.plot(history["epoch"], history["train_loss"], label="train_loss")
    if "val_loss" in history.columns:
        ax.plot(history["epoch"], history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: Dict[str, Any], out_path: Path, title: str = "Confusion Matrix") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = np.asarray(cm["matrix"] if isinstance(cm, dict) and "matrix" in cm else cm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Legit", "Pred Fraud"],
        yticklabels=["True Legit", "True Fraud"],
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_pr(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    roc_path: Path,
    pr_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    roc_path.parent.mkdir(parents=True, exist_ok=True)

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(y_true, y_prob):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, label=f"PR-AUC={average_precision_score(y_true, y_prob):.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)


def plot_class_distribution(splits: Dict[str, pd.DataFrame], out_path: Path, label_col: str = "label") -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    names, legit, fraud = [], [], []
    for name, df in splits.items():
        names.append(name)
        legit.append(int((df[label_col] == 0).sum()))
        fraud.append(int((df[label_col] == 1).sum()))
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, legit, width, label="Legitimate")
    ax.bar(x + width / 2, fraud, width, label="Fraudulent")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution by Split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(comparison_csv: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not comparison_csv.exists():
        return
    df = pd.read_csv(comparison_csv)
    if df.empty:
        return
    # Drop unavailable rows from bar chart values but keep labels.
    plot_df = df.copy()
    for col in ["Fraud Recall", "Fraud F1", "PR-AUC"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    labels = [
        f"{m}\n({i})"
        for m, i in zip(plot_df["Model"], plot_df["Imbalance method"], strict=True)
    ]
    x = np.arange(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.2), 5))
    ax.bar(x - width, plot_df["Fraud Recall"], width, label="Fraud Recall")
    ax.bar(x, plot_df["Fraud F1"], width, label="Fraud F1")
    ax.bar(x + width, plot_df["PR-AUC"], width, label="PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison (fraud-focused)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ========================================================================
# DATASET
# ========================================================================

SPLIT_FILES = {
    "train": TRAIN_CSV,
    "validation": VALIDATION_CSV,
    "test": TEST_CSV,
}


def load_split(split: str, path: Optional[Path] = None) -> pd.DataFrame:
    """Load one fixed split CSV. Does not merge or re-partition."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_FILES)}")
    csv_path = Path(path) if path is not None else SPLIT_FILES[split]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Required split file missing: {csv_path}. "
            "Do not create a new split; use the project train/validation/test CSVs."
        )
    df = pd.read_csv(csv_path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{csv_path} missing required column '{LABEL_COLUMN}'")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    # Paper-aligned splits already store the exact model_text used in training.
    # Do not rebuild it — retagging would change the input and break reproduction.
    if "model_text" not in df.columns or df["model_text"].isna().all():
        df = prepare_dataframe_text(df)
    else:
        df["model_text"] = df["model_text"].fillna("").astype(str)
    df["split"] = split
    df["row_index"] = np.arange(len(df), dtype=np.int64)
    if ID_COLUMN not in df.columns:
        df[ID_COLUMN] = [f"{split}_{i:05d}" for i in range(len(df))]
    if "title" not in df.columns:
        df["title"] = df["model_text"].map(derive_title_preview)
    if "company" not in df.columns:
        # Splits do not store company separately; derive a preview for analysis.
        df["company"] = df["model_text"].map(derive_company_preview)
    return df


def load_all_splits(
    train_path: Optional[Path] = None,
    validation_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Load the three fixed splits independently (no merging)."""
    return {
        "train": load_split("train", train_path),
        "validation": load_split("validation", validation_path),
        "test": load_split("test", test_path),
    }


def _label_stats(df: pd.DataFrame) -> Dict[str, Any]:
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    n0 = int(counts.get(0, 0))
    n1 = int(counts.get(1, 0))
    total = len(df)
    return {
        "n_samples": total,
        "n_legitimate": n0,
        "n_fraudulent": n1,
        "class_ratio_legitimate": float(n0 / total) if total else 0.0,
        "class_ratio_fraudulent": float(n1 / total) if total else 0.0,
        "label_counts": {str(k): int(v) for k, v in sorted(counts.items())},
    }


def _text_stats(df: pd.DataFrame) -> Dict[str, Any]:
    texts = df["model_text"].fillna("")
    lengths = texts.str.len()
    return {
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "empty_text_count": int((texts.str.strip() == "").sum()),
        "exact_duplicate_text_count": int(texts.duplicated().sum()),
        "duplicate_id_count": int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df.columns else None,
        "text_length": {
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "median": float(lengths.median()) if len(lengths) else 0.0,
            "min": int(lengths.min()) if len(lengths) else 0,
            "max": int(lengths.max()) if len(lengths) else 0,
            "p90": float(lengths.quantile(0.90)) if len(lengths) else 0.0,
            "p95": float(lengths.quantile(0.95)) if len(lengths) else 0.0,
            "p99": float(lengths.quantile(0.99)) if len(lengths) else 0.0,
        },
        "columns": list(df.columns),
        "title_unique": int(df["title"].nunique()) if "title" in df.columns else None,
        "company_unique": int(df["company"].nunique()) if "company" in df.columns else None,
    }


def compute_token_length_stats(
    texts: Sequence[str],
    tokenizer,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Tokenise without truncation to estimate length distribution."""
    sample = list(texts)
    if max_samples is not None and len(sample) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(sample), size=max_samples, replace=False)
        sample = [sample[i] for i in idx]

    lengths: List[int] = []
    for text in sample:
        enc = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        lengths.append(len(enc["input_ids"]))

    arr = np.asarray(lengths, dtype=np.float64)

    def pct_over(limit: int) -> float:
        return float((arr > limit).mean()) if len(arr) else 0.0

    p_over_256 = pct_over(256)
    p_over_384 = pct_over(384)
    p_over_512 = pct_over(512)
    reason = (
        f"Measured train token lengths: median≈{float(np.median(arr)) if len(arr) else 0:.0f}, "
        f"{p_over_256:.1%} >256, {p_over_384:.1%} >384, {p_over_512:.1%} >512. "
        "Default max_length=384 is chosen for RTX 3060 with batch=4 / accum=4; "
        "try 512 only after an OOM check. Long ads may still be truncated."
    )
    return {
        "n_texts_measured": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
        "pct_over_128": pct_over(128),
        "pct_over_256": p_over_256,
        "pct_over_384": p_over_384,
        "pct_over_512": p_over_512,
        "recommended_max_length": 384,
        "recommendation_reason": reason,
    }


def build_data_quality_report(
    splits: Dict[str, pd.DataFrame],
    token_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce a report only — never re-split based on findings."""
    per_split = {}
    for name, df in splits.items():
        per_split[name] = {**_label_stats(df), **_text_stats(df)}

    train, val, test = splits["train"], splits["validation"], splits["test"]

    def set_overlap(a: pd.Series, b: pd.Series) -> int:
        return int(len(set(a.astype(str)) & set(b.astype(str))))

    cross = {
        "identical_text_train_validation": set_overlap(train["model_text"], val["model_text"]),
        "identical_text_train_test": set_overlap(train["model_text"], test["model_text"]),
        "identical_text_validation_test": set_overlap(val["model_text"], test["model_text"]),
        "identical_id_train_validation": set_overlap(train[ID_COLUMN], val[ID_COLUMN]),
        "identical_id_train_test": set_overlap(train[ID_COLUMN], test[ID_COLUMN]),
        "identical_id_validation_test": set_overlap(val[ID_COLUMN], test[ID_COLUMN]),
        "note": (
            "Cross-split overlaps are reported for awareness only. "
            "This module never reassigns rows across splits."
        ),
    }

    if GROUP_COLUMN in train.columns:
        cross.update(
            {
                "group_overlap_train_validation": set_overlap(train[GROUP_COLUMN], val[GROUP_COLUMN]),
                "group_overlap_train_test": set_overlap(train[GROUP_COLUMN], test[GROUP_COLUMN]),
                "group_overlap_validation_test": set_overlap(val[GROUP_COLUMN], test[GROUP_COLUMN]),
            }
        )

    # Company+title combo check when both derived/present.
    if "company" in train.columns and "title" in train.columns:
        def combo(df: pd.DataFrame) -> pd.Series:
            return df["company"].astype(str) + " || " + df["title"].astype(str)

        cross["company_title_overlap_train_validation"] = set_overlap(combo(train), combo(val))
        cross["company_title_overlap_train_test"] = set_overlap(combo(train), combo(test))
        cross["company_title_overlap_validation_test"] = set_overlap(combo(val), combo(test))

    report: Dict[str, Any] = {
        "split_files": {k: str(SPLIT_FILES[k]) for k in SPLIT_FILES},
        "per_split": per_split,
        "cross_split_checks": cross,
        "policy": {
            "no_resplit": True,
            "no_merge": True,
            "no_cross_validation": True,
            "oversample_train_only": True,
        },
    }
    if token_stats is not None:
        report["token_length_stats_train"] = token_stats
    return report


def compute_class_weights(
    y: np.ndarray,
    n_classes: int = 2,
    *,
    transform: str = "none",
    weight_max: Optional[float] = None,
) -> np.ndarray:
    """Inverse-frequency weights from TRAIN labels only.

    weight[c] = N / (n_classes * count[c])

    Optional softening:
      - sqrt: square-root of inverse-frequency weights
      - sqrt_clip: sqrt then clip to [1/weight_max, weight_max] (or [0, weight_max] if max set)
      - clip: clip raw inverse-frequency to weight_max
      - none: raw inverse-frequency
    """
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    total = float(len(y))
    weights = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        if counts[c] > 0:
            weights[c] = total / (n_classes * counts[c])
        else:
            weights[c] = 1.0

    transform = (transform or "none").strip().lower()
    if transform in {"sqrt", "sqrt_clip"}:
        weights = np.sqrt(weights)
    if transform in {"clip", "sqrt_clip"} and weight_max is not None and weight_max > 0:
        weights = np.clip(weights, 1.0 / float(weight_max), float(weight_max))
    return weights


def oversample_fraud_train(
    df: pd.DataFrame,
    *,
    factor: float = 3.0,
    seed: int = 42,
    label_column: str = LABEL_COLUMN,
    fraud_label: int = 1,
) -> pd.DataFrame:
    """Train-only minority oversampling. Validation/test must never be passed here.

    ``factor`` multiplies the fraud row count (e.g. 3.0 => ~3x fraud examples).
    """
    if factor is None or float(factor) <= 1.0:
        return df.reset_index(drop=True)

    factor = float(factor)
    fraud = df[df[label_column] == fraud_label]
    if fraud.empty:
        return df.reset_index(drop=True)

    n_target = int(round(len(fraud) * factor))
    extra = n_target - len(fraud)
    if extra <= 0:
        return df.reset_index(drop=True)

    rng = np.random.RandomState(int(seed))
    sampled = fraud.sample(n=extra, replace=True, random_state=rng)
    out = pd.concat([df, sampled], ignore_index=True)
    return out.sample(frac=1.0, random_state=rng).reset_index(drop=True)


class JobTextDataset(Dataset):
    """Tokenised job-ad dataset for end-to-end BERT fine-tuning."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 256,
        record_ids: Optional[Sequence[str]] = None,
        titles: Optional[Sequence[str]] = None,
        companies: Optional[Sequence[str]] = None,
    ) -> None:
        self.texts = list(texts)
        self.labels = [int(x) for x in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.record_ids = list(record_ids) if record_ids is not None else [str(i) for i in range(len(self.texts))]
        self.titles = list(titles) if titles is not None else [""] * len(self.texts)
        self.companies = list(companies) if companies is not None else [""] * len(self.texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.labels[idx],
            "record_id": self.record_ids[idx],
            "title": self.titles[idx],
            "company": self.companies[idx],
            "row_index": idx,
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"]
        return item


def dataframe_to_text_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> JobTextDataset:
    return JobTextDataset(
        texts=df["model_text"].tolist(),
        labels=df[LABEL_COLUMN].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        record_ids=df[ID_COLUMN].astype(str).tolist(),
        titles=df["title"].astype(str).tolist() if "title" in df.columns else None,
        companies=df["company"].astype(str).tolist() if "company" in df.columns else None,
    )

