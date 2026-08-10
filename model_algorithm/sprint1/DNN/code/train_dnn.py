"""Train a TF-IDF + MLP (DNN) classifier on sprint1 70/15/15 splits.

- Fit TF-IDF on train only
- Train a small feed-forward net with PyTorch (CUDA if available)
- Choose decision threshold on validation by max Fraud F1
- Evaluate once on test
- Save: result/test_metrics.json (fraud F1 / Recall / Precision)
         weight/dnn_mlp.pt, tfidf.joblib, threshold.json
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "data" / "splits"
RESULT_DIR = ROOT / "result"
WEIGHT_DIR = ROOT / "weight"

SEED = 42
MAX_FEATURES = 30_000
HIDDEN = 256
DROPOUT = 0.3
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
PATIENCE = 5


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    df = pd.read_csv(path, usecols=["record_id", "label", "combined_text"])
    df["combined_text"] = df["combined_text"].fillna("").astype(str)
    return df


class FraudMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def to_dense_tensor(matrix) -> torch.Tensor:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return torch.tensor(matrix, dtype=torch.float32)


@torch.no_grad()
def predict_scores(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(x), batch_size=BATCH_SIZE, shuffle=False)
    outs = []
    for (batch,) in loader:
        logits = model(batch.to(device))
        outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs, axis=0)


def select_fraud_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    best = float(np.max(f1_values))
    idx = int(np.flatnonzero(np.isclose(f1_values, best))[0])
    return float(thresholds[idx])


def fraud_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    return {
        "fraud_precision": float(precision_score(labels, preds, zero_division=0)),
        "fraud_recall": float(recall_score(labels, preds, zero_division=0)),
        "fraud_f1": float(f1_score(labels, preds, zero_division=0)),
        "threshold": float(threshold),
    }


def main() -> None:
    set_seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=2,
        max_df=0.98,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    x_train = vectorizer.fit_transform(train["combined_text"])
    x_val = vectorizer.transform(validation["combined_text"])
    x_test = vectorizer.transform(test["combined_text"])

    y_train = train["label"].to_numpy().astype(np.float32)
    y_val = validation["label"].to_numpy().astype(np.float32)
    y_test = test["label"].to_numpy().astype(np.float32)

    x_train_t = to_dense_tensor(x_train)
    x_val_t = to_dense_tensor(x_val)
    x_test_t = to_dense_tensor(x_test)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    # Positive class weight for imbalance.
    n_pos = max(float(y_train.sum()), 1.0)
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    model = FraudMLP(n_features=x_train_t.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    best_val_f1 = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)

        val_scores = predict_scores(model, x_val_t, device)
        thr = select_fraud_f1_threshold(y_val, val_scores)
        val_f1 = fraud_metrics(y_val, val_scores, thr)["fraud_f1"]
        avg_loss = total_loss / len(y_train)
        print(
            f"epoch={epoch:02d} loss={avg_loss:.4f} "
            f"val_fraud_f1={val_f1:.4f} thr={thr:.4f}"
        )

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best state saved")
    model.load_state_dict(best_state)
    model.to(device)

    val_scores = predict_scores(model, x_val_t, device)
    threshold = select_fraud_f1_threshold(y_val, val_scores)
    test_scores = predict_scores(model, x_test_t, device)
    metrics = fraud_metrics(y_test, test_scores, threshold)
    result = {
        "fraud_f1": metrics["fraud_f1"],
        "fraud_recall": metrics["fraud_recall"],
        "fraud_precision": metrics["fraud_precision"],
    }

    (RESULT_DIR / "test_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "n_features": int(x_train_t.shape[1]),
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "threshold": float(threshold),
        },
        WEIGHT_DIR / "dnn_mlp.pt",
    )
    joblib.dump(vectorizer, WEIGHT_DIR / "tfidf.joblib")

    print(json.dumps(result, indent=2))
    print(f"best_val_fraud_f1={best_val_f1:.4f} threshold={threshold:.4f} device={device}")
    print(f"Saved weights under: {WEIGHT_DIR}")
    print(f"Saved result: {RESULT_DIR / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
