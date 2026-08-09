"""Shared HuggingFace transformer helpers for BERT / RoBERTa pipelines."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


class JobTextDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: np.ndarray | None = None):
        self.encodings = encodings
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.encodings["input_ids"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_texts(tokenizer, texts: list[str] | pd.Series, max_len: int) -> dict[str, torch.Tensor]:
    cleaned = [str(t) if t is not None else "" for t in texts]
    encoded = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return dict(encoded.items())


def _class_weight_tensor(class_weight: dict[int, float], device: torch.device) -> torch.Tensor:
    weights = [float(class_weight.get(0, 1.0)), float(class_weight.get(1, 1.0))]
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_proba(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str] | pd.Series,
    *,
    max_len: int,
    batch_size: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return P(fake) for each text."""
    device = device or resolve_device()
    model.eval()
    dataset = JobTextDataset(encode_texts(tokenizer, texts, max_len))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs: list[np.ndarray] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        batch_probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs.append(batch_probs)
    if not probs:
        return np.array([], dtype=np.float64)
    return np.concatenate(probs, axis=0)


def train_transformer_classifier(
    *,
    pretrained_model_name: str,
    train_texts: pd.Series,
    train_labels: np.ndarray,
    val_texts: pd.Series,
    val_labels: np.ndarray,
    saved_model_dir: Path,
    max_len: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    class_weight: dict[int, float],
    early_stopping_patience: int,
    seed: int,
    warmup_ratio: float = 0.1,
    meta_extra: dict[str, Any] | None = None,
) -> Path:
    """Fine-tune a sequence classifier and save model + tokenizer under saved_model_dir."""
    set_seed(seed)
    device = resolve_device()
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=2,
        id2label={0: "real", 1: "fake"},
        label2id={"real": 0, "fake": 1},
    )
    model.to(device)

    train_dataset = JobTextDataset(
        encode_texts(tokenizer, train_texts, max_len),
        train_labels.astype(np.int64),
    )
    val_dataset = JobTextDataset(
        encode_texts(tokenizer, val_texts, max_len),
        val_labels.astype(np.int64),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = max(0, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=_class_weight_tensor(class_weight, device))

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = early_stopping_patience

    print(f"Device: {device}")
    print(f"Train batches/epoch: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)
                val_loss += float(loss.item())

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch}/{epochs} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.save_pretrained(saved_model_dir)
    tokenizer.save_pretrained(saved_model_dir)

    meta = {
        "pretrained_model_name": pretrained_model_name,
        "max_len": max_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "class_weight": {str(k): v for k, v in class_weight.items()},
        "best_val_loss": best_val_loss,
        "device_used": str(device),
    }
    if meta_extra:
        meta.update(meta_extra)
    (saved_model_dir / "pipeline_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Model saved: {saved_model_dir}")
    return saved_model_dir


def load_transformer_artifacts(saved_model_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Load fine-tuned model, tokenizer, and pipeline_config.json."""
    config_path = saved_model_dir / "pipeline_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing pipeline_config.json under {saved_model_dir}")
    if not (saved_model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Transformer model not found under {saved_model_dir}. Run train_model.py first."
        )

    meta = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_dir)
    model.to(resolve_device())
    model.eval()
    return model, tokenizer, meta
