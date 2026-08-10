"""BERT inference service (paper-aligned maxlen=512 checkpoint)."""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

import torch
from transformers import AutoTokenizer

from settings import BERT_CHECKPOINT, BERT_CODE_DIR, BERT_MAX_LENGTH, load_runtime_config


class BERTService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._device: torch.device | None = None
        self._model_threshold: float | None = None

    def load(self, allow_cpu: bool | None = None) -> None:
        if allow_cpu is None:
            allow_cpu = os.getenv("ALLOW_CPU", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
        with self._lock:
            if self._model is not None:
                return
            if not BERT_CHECKPOINT.exists():
                raise FileNotFoundError(f"BERT checkpoint missing: {BERT_CHECKPOINT}")

            # Prefer BERT package modules over any same-named API modules.
            bert_code = str(BERT_CODE_DIR)
            if sys.path[:1] != [bert_code]:
                sys.path = [bert_code] + [p for p in sys.path if p != bert_code]

            from model import BertForFraudClassification, softmax_fraud_proba  # noqa: WPS433
            from utils import get_device, load_json, setup_logging  # noqa: WPS433
            import torch.nn as nn  # noqa: WPS433
            from transformers import AutoModelForSequenceClassification  # noqa: WPS433

            self._softmax_fraud_proba = softmax_fraud_proba
            logger = setup_logging("api_bert.log")
            self._device = get_device(allow_cpu=bool(allow_cpu), logger=logger)
            tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT)

            # Load fine-tuned weights directly (no bert-base-uncased re-init).
            wrapper = BertForFraudClassification.__new__(BertForFraudClassification)
            nn.Module.__init__(wrapper)
            wrapper.model = AutoModelForSequenceClassification.from_pretrained(
                BERT_CHECKPOINT
            )
            wrapper = wrapper.to(self._device)
            wrapper.eval()

            thr_path = BERT_CHECKPOINT / "threshold.json"
            threshold = 0.5
            if thr_path.exists():
                threshold = float(load_json(thr_path)["threshold"])
            runtime = load_runtime_config()["bert"]["model_threshold"]
            if runtime is not None:
                threshold = float(runtime)

            self._model = wrapper
            self._tokenizer = tokenizer
            self._model_threshold = threshold

    @property
    def model_threshold(self) -> float:
        self.load()
        assert self._model_threshold is not None
        return self._model_threshold

    @torch.no_grad()
    def predict(self, model_text: str, threshold: float | None = None) -> dict[str, Any]:
        self.load()
        assert self._model is not None and self._tokenizer is not None
        assert self._device is not None and self._model_threshold is not None

        thr = float(self._model_threshold if threshold is None else threshold)
        enc = self._tokenizer(
            model_text,
            truncation=True,
            max_length=BERT_MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        logits = self._model(**enc).logits
        score = float(self._softmax_fraud_proba(logits)[0].cpu().item())
        pred = int(score >= thr)
        return {
            "model": "bert",
            "bert_score": score,
            "threshold": thr,
            "max_length": BERT_MAX_LENGTH,
            "predicted_label_id": pred,
            "predicted_label": "Fraudulent" if pred == 1 else "Legitimate",
            "device": str(self._device),
        }


bert_service = BERTService()
