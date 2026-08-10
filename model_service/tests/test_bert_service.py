from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from services.bert_service import BERTService


def test_bert_service_missing_checkpoint_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    missing_checkpoint = tmp_path / "missing"
    monkeypatch.setattr("services.bert_service.BERT_CHECKPOINT", missing_checkpoint)

    with pytest.raises(FileNotFoundError, match="BERT checkpoint missing"):
        BERTService().load()


def test_bert_service_rejects_cpu_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("services.bert_service.BERT_CHECKPOINT", tmp_path)
    monkeypatch.setattr("services.bert_service.torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CPU inference is disabled"):
        BERTService().load(allow_cpu=False)


def test_bert_service_loads_and_predicts_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "threshold.json").write_text('{"threshold": 0.4}', encoding="utf-8")
    monkeypatch.setattr("services.bert_service.BERT_CHECKPOINT", tmp_path)
    monkeypatch.setattr("services.bert_service.BERT_MAX_LENGTH", 128)
    monkeypatch.setattr("services.bert_service.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "services.bert_service.load_runtime_config",
        lambda: {"bert": {"model_threshold": 0.7}},
    )

    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    monkeypatch.setattr(
        "services.bert_service.AutoTokenizer.from_pretrained",
        lambda _checkpoint: tokenizer,
    )

    model = MagicMock()
    model.to.return_value = model
    model.return_value.logits = torch.tensor([[0.0, 2.0]])
    monkeypatch.setattr(
        "services.bert_service.AutoModelForSequenceClassification.from_pretrained",
        lambda _checkpoint: model,
    )

    service = BERTService()
    result = service.predict("sample job", threshold=0.8)

    assert service.model_threshold == pytest.approx(0.7)
    assert result["bert_score"] == pytest.approx(0.880797, rel=1e-5)
    assert result["threshold"] == pytest.approx(0.8)
    assert result["max_length"] == 128
    assert result["predicted_label_id"] == 1
    assert result["predicted_label"] == "Fraudulent"
    assert result["device"] == "cpu"
    tokenizer.assert_called_once_with(
        "sample job",
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )
    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once_with()

