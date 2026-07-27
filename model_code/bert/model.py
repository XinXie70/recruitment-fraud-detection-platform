"""BERT classifier for fraudulent job-ad detection."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification


class BertForFraudClassification(nn.Module):
    """Thin wrapper around AutoModelForSequenceClassification."""

    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int = 2,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = dropout
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            config.classifier_dropout = dropout
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=config,
        )
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return self.model(**kwargs)

    @property
    def encoder(self):
        return self.model.bert if hasattr(self.model, "bert") else self.model.base_model


def softmax_fraud_proba(logits):
    """Return P(fraud=1) from 2-class logits."""
    return F.softmax(logits, dim=-1)[:, 1]
