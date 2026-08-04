"""Model loading and inference for LR, BERT, and LR+BERT ensemble."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BERT_CODE_DIR = PROJECT_ROOT / "model_code" / "bert"
if str(BERT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(BERT_CODE_DIR))

from model import BertForFraudClassification, softmax_fraud_proba  # noqa: E402
from preprocessing import clean_text  # noqa: E402
from utils import get_device, load_json  # noqa: E402

LR_ARTIFACT = PROJECT_ROOT / "model_weights" / "logistic_regression" / "logistic_regression_baseline.joblib"
LR_METRICS = PROJECT_ROOT / "reports" / "models" / "logistic_regression" / "validation_metrics.json"
BERT_CHECKPOINT = PROJECT_ROOT / "model_weights" / "bert" / "bert_class_weighted" / "best"
ENSEMBLE_CONFIG = PROJECT_ROOT / "reports" / "models" / "ensemble_lr_bert" / "ensemble_config.json"


class ModelService:
    """Lazy-load LR and BERT once, then serve ensemble and risk scoring."""

    def __init__(self, allow_cpu: bool = True) -> None:
        self.allow_cpu = allow_cpu
        self._lr_model = None
        self._lr_threshold: float | None = None
        self._bert_model = None
        self._bert_tokenizer = None
        self._bert_threshold: float | None = None
        self._device: torch.device | None = None
        self._ensemble_config: dict | None = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = get_device(allow_cpu=self.allow_cpu, logger=None)
        return self._device

    def load_lr(self) -> None:
        if self._lr_model is not None:
            return
        if not LR_ARTIFACT.exists():
            raise FileNotFoundError(
                f"LR artifact missing: {LR_ARTIFACT}. "
                "Run: python src/models/logistic_regression/train_baseline.py"
            )
        metrics = json.loads(LR_METRICS.read_text(encoding="utf-8"))
        self._lr_threshold = float(metrics["selected_threshold"]["threshold"])
        self._lr_model = joblib.load(LR_ARTIFACT)

    def load_bert(self) -> None:
        if self._bert_model is not None:
            return
        if not BERT_CHECKPOINT.exists():
            raise FileNotFoundError(f"BERT checkpoint missing: {BERT_CHECKPOINT}")
        self._bert_tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT)
        wrapper = BertForFraudClassification("bert-base-uncased")
        wrapper.model = type(wrapper.model).from_pretrained(BERT_CHECKPOINT)
        self._bert_model = wrapper.to(self.device)
        self._bert_model.eval()
        thr_path = BERT_CHECKPOINT / "threshold.json"
        self._bert_threshold = 0.5
        if thr_path.exists():
            self._bert_threshold = float(load_json(thr_path).get("threshold", 0.5))

    def load_ensemble_config(self) -> dict:
        if self._ensemble_config is None:
            if not ENSEMBLE_CONFIG.exists():
                raise FileNotFoundError(f"Ensemble config missing: {ENSEMBLE_CONFIG}")
            self._ensemble_config = json.loads(ENSEMBLE_CONFIG.read_text(encoding="utf-8"))
        return self._ensemble_config

    def predict_lr(self, text: str) -> dict[str, Any]:
        self.load_lr()
        cleaned = clean_text(text)
        score = float(self._lr_model.predict_proba([cleaned])[0, 1])
        threshold = self._lr_threshold
        prediction = 1 if score >= threshold else 0
        return {
            "model": "logistic_regression_baseline",
            "fraud_score": score,
            "threshold": threshold,
            "prediction": prediction,
            "predicted_label": "Fraudulent" if prediction == 1 else "Legitimate",
        }

    @torch.no_grad()
    def predict_bert(self, text: str, max_length: int = 256) -> dict[str, Any]:
        self.load_bert()
        cleaned = clean_text(text)
        enc = self._bert_tokenizer(
            cleaned,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self._bert_model(**enc).logits
        score = float(softmax_fraud_proba(logits)[0].cpu().item())
        threshold = self._bert_threshold
        prediction = 1 if score >= threshold else 0
        return {
            "model": "bert_class_weighted",
            "fraud_score": score,
            "threshold": threshold,
            "prediction": prediction,
            "predicted_label": "Fraudulent" if prediction == 1 else "Legitimate",
        }

    def predict_ensemble(self, text: str) -> dict[str, Any]:
        lr = self.predict_lr(text)
        bert = self.predict_bert(text)
        config = self.load_ensemble_config()
        w_lr = float(config["weights"]["logistic_regression_baseline"])
        w_bert = float(config["weights"]["bert_class_weighted"])
        score = w_lr * lr["fraud_score"] + w_bert * bert["fraud_score"]
        threshold = float(config["selected_threshold"])
        prediction = 1 if score >= threshold else 0
        return {
            "model": "ensemble_lr_bert_class_weighted",
            "fraud_score": score,
            "threshold": threshold,
            "prediction": prediction,
            "predicted_label": "Fraudulent" if prediction == 1 else "Legitimate",
            "weights": {
                "logistic_regression_baseline": w_lr,
                "bert_class_weighted": w_bert,
            },
            "lr_fraud_score": lr["fraud_score"],
            "bert_fraud_score": bert["fraud_score"],
        }

    def models_loaded(self) -> dict[str, bool]:
        return {
            "logistic_regression": self._lr_model is not None,
            "bert": self._bert_model is not None,
        }
