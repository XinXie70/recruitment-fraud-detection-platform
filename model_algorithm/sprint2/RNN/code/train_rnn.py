"""Train a unidirectional RNN text classifier on sprint1 splits."""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent.parent / "sprint1" / "data" / "splits"
RESULT_DIR = ROOT / "result"
WEIGHT_DIR = ROOT / "weight"

SEED = 42
MAX_LEN = 200
MIN_FREQ = 2
MAX_VOCAB = 20_000
EMBED_DIM = 128
HIDDEN = 128
BATCH_SIZE = 64
EPOCHS = 12
LR = 1e-3
PATIENCE = 4
PAD_IDX = 0
UNK_IDX = 1


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.csv", usecols=["label", "combined_text"])
    df["combined_text"] = df["combined_text"].fillna("").astype(str)
    return df


def build_vocab(texts: list[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    most_common = [w for w, c in counter.most_common(MAX_VOCAB) if c >= MIN_FREQ]
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for word in most_common:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int]) -> torch.Tensor:
    ids = [vocab.get(tok, UNK_IDX) for tok in tokenize(text)[:MAX_LEN]]
    if not ids:
        ids = [UNK_IDX]
    return torch.tensor(ids, dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, vocab: dict[str, int]):
        self.seqs = [encode(t, vocab) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.seqs[idx], self.labels[idx]


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    return padded, lengths, torch.stack(labels)


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.rnn = nn.RNN(embed_dim, hidden, batch_first=True, nonlinearity="tanh")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        out = self.dropout(h_n[-1])
        return self.fc(out).squeeze(-1)


@torch.no_grad()
def predict_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for xb, lengths, yb in loader:
        logits = model(xb.to(device), lengths)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = float(np.max(f1_values))
    idx = int(np.flatnonzero(np.isclose(f1_values, best))[0])
    return float(thresholds[idx])


def fraud_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    return {
        "fraud_f1": float(f1_score(labels, preds, zero_division=0)),
        "fraud_recall": float(recall_score(labels, preds, zero_division=0)),
        "fraud_precision": float(precision_score(labels, preds, zero_division=0)),
    }


def main() -> None:
    set_seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train, validation, test = load_split("train"), load_split("validation"), load_split("test")
    vocab = build_vocab(train["combined_text"].tolist())

    train_ds = TextDataset(train["combined_text"].tolist(), train["label"].to_numpy(), vocab)
    val_ds = TextDataset(validation["combined_text"].tolist(), validation["label"].to_numpy(), vocab)
    test_ds = TextDataset(test["combined_text"].tolist(), test["label"].to_numpy(), vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    y_train = train["label"].to_numpy()
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)], device=device)

    model = RNNClassifier(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1, best_state, wait = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, lengths, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, lengths), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)

        val_scores, val_labels = predict_scores(model, val_loader, device)
        thr = select_threshold(val_labels, val_scores)
        val_f1 = fraud_metrics(val_labels, val_scores, thr)["fraud_f1"]
        print(f"epoch={epoch:02d} loss={total_loss/len(train_ds):.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    val_scores, val_labels = predict_scores(model, val_loader, device)
    threshold = select_threshold(val_labels, val_scores)
    test_scores, test_labels = predict_scores(model, test_loader, device)
    result = fraud_metrics(test_labels, test_scores, threshold)

    (RESULT_DIR / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": best_state,
            "vocab": vocab,
            "threshold": float(threshold),
            "max_len": MAX_LEN,
            "embed_dim": EMBED_DIM,
            "hidden": HIDDEN,
        },
        WEIGHT_DIR / "rnn.pt",
    )
    print(json.dumps(result, indent=2))
    print(f"best_val_f1={best_val_f1:.4f} threshold={threshold:.4f}")


if __name__ == "__main__":
    main()
