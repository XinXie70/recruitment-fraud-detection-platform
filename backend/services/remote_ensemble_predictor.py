from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from backend.schemas.analysis import EnsembleResult, ModelMemberOutput
from backend.services.ensemble_predictor import EnsembleComputation, EnsembleUnavailableError


_RISK_LEVELS = {"Low": "low", "Suspicious": "medium", "High": "high"}
_DISPLAY_NAMES = {"lr": "Logistic Regression", "bert": "BERT"}
_NEW_API_ENDPOINTS = {
    "lr": "/predict/post_predict_lr",
    "bert": "/predict/post_predict_bert",
}
_LEGACY_API_ENDPOINTS = {
    "lr": "/predict/lr",
    "bert": "/predict/bert",
}


class RemoteFinalEnsemblePredictor:
    """Adapter for the deployed LR + BERT FP-gate model API."""

    def __init__(self, base_url: str, timeout_seconds: float = 120.0):
        self.raw_base_url = base_url
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_seconds)

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = self.client.post(f"{self.base_url}{path}", json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EnsembleUnavailableError(
                "Remote final ensemble request failed."
            ) from exc
        if not isinstance(data, dict):
            raise EnsembleUnavailableError("Remote final ensemble returned an error.")
        if isinstance(data.get("ok"), bool) and data.get("ok") is False:
            raise EnsembleUnavailableError("Remote final ensemble returned an error.")
        return data

    @staticmethod
    def _probability(value: str | int | float, field: str) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise EnsembleUnavailableError(
                f"Remote final ensemble returned invalid {field}."
            ) from exc
        if not 0 <= score <= 1:
            raise EnsembleUnavailableError(
                f"Remote final ensemble returned invalid {field}."
            )
        return score

    def warm_up(self, sample: str) -> dict[str, str | None]:
        try:
            self.predict(sample)
        except EnsembleUnavailableError as exc:
            return {"final_ensemble": str(exc)}
        return {"final_ensemble": None}

    def _extract_probability(self, payload: dict[str, Any], field: str) -> float:
        if isinstance(payload, dict) and field in payload:
            return self._probability(payload[field], field)
        raise EnsembleUnavailableError(
            "Remote final ensemble response does not match its API contract."
        )

    def _try_new_api(self, text: str) -> tuple[float, float, dict[str, Any]]:
        lr_response = self._post(_NEW_API_ENDPOINTS["lr"], {"text": text})
        bert_response = self._post(_NEW_API_ENDPOINTS["bert"], {"text": text})
        lr_score = self._extract_probability(lr_response, "fraud_score")
        bert_score = self._extract_probability(bert_response, "fraud_score")
        risk = {
            "risk_score": max(lr_score, bert_score),
            "risk_level": "High" if max(lr_score, bert_score) >= 0.5 else "Low",
            "risk_score_source": "bert",
            "gate_triggered": False,
            "decision_reason": "New API fallback: used direct BERT score",
            "thresholds": {
                "bert_low_threshold": 0.15,
                "bert_high_threshold": 0.32,
                "lr_gate": 0.2,
            },
        }
        return lr_score, bert_score, risk

    def _try_legacy_api(self, text: str) -> tuple[float, float, dict[str, Any]]:
        data = self._post("/predict/all", {"text": text})
        try:
            lr_score = self._probability(data["lr"]["lr_score"], "lr_score")
            bert_score = self._probability(
                data["bert"]["bert_score"], "bert_score"
            )
            risk = data["risk"]
        except (KeyError, TypeError) as exc:
            raise EnsembleUnavailableError(
                "Remote final ensemble response does not match its API contract."
            ) from exc
        return lr_score, bert_score, risk

    def predict(self, text: str) -> EnsembleComputation:
        if self.raw_base_url.endswith("/"):
            try:
                lr_score, bert_score, risk = self._try_legacy_api(text)
            except EnsembleUnavailableError as exc:
                if str(exc) != "Remote final ensemble response does not match its API contract.":
                    raise
                lr_score, bert_score, risk = self._try_new_api(text)
        else:
            try:
                lr_score, bert_score, risk = self._try_new_api(text)
            except EnsembleUnavailableError as exc:
                if str(exc) != "Remote final ensemble response does not match its API contract.":
                    raise
                lr_score, bert_score, risk = self._try_legacy_api(text)
        try:
            risk_score = self._probability(risk["risk_score"], "risk_score")
            risk_level = _RISK_LEVELS[risk["risk_level"]]
            thresholds = risk["thresholds"]
            low_threshold = self._probability(
                thresholds["bert_low_threshold"], "bert_low_threshold"
            )
            high_threshold = self._probability(
                thresholds["bert_high_threshold"], "bert_high_threshold"
            )
            lr_gate_threshold = self._probability(
                thresholds["lr_gate"], "lr_gate"
            )
            source = str(risk["risk_score_source"])
            if source not in {"bert", "lr_gate"}:
                raise TypeError
            gate_triggered = risk["gate_triggered"]
            if not isinstance(gate_triggered, bool):
                raise TypeError
            decision_reason = str(risk["decision_reason"])
        except (KeyError, TypeError) as exc:
            raise EnsembleUnavailableError(
                "Remote final ensemble response does not match its API contract."
            ) from exc

        labels = {
            "low": ("Likely Legitimate", "real", "Safe"),
            "medium": ("Suspicious", "fake", "Review Required"),
            "high": ("Likely Deceptive", "fake", "High Risk Warning"),
        }
        classification, prediction, action = labels[risk_level]
        scores = {"lr": lr_score, "bert": bert_score}
        roles = {
            "lr": "false_positive_gate",
            "bert": "primary_score",
        }
        members = [
            ModelMemberOutput(
                key=key,
                display_name=_DISPLAY_NAMES[key],
                status="success",
                raw_score=score,
                calibrated_score=score,
                role=roles[key],
                decision_active=source.startswith(key),
            )
            for key, score in scores.items()
        ]
        ensemble = EnsembleResult(
            status="success",
            risk_score=risk_score,
            classification_label=classification,
            risk_level=risk_level,
            prediction=prediction,
            recommended_action=action,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            active_model_count=2,
            failed_model_count=0,
            version="remote-lr-bert-fp-gate-v1",
            fitted=True,
            weight_source="remote_fp_gate",
            method="bert_lr_fp_gate",
            risk_score_source=source,
            gate_triggered=gate_triggered,
            decision_reason=decision_reason,
            bert_low_threshold=low_threshold,
            bert_high_threshold=high_threshold,
            lr_gate_threshold=lr_gate_threshold,
        )

        def score_batch(texts: Sequence[str]) -> list[float]:
            values = list(texts)
            scores: list[float] = []
            for start in range(0, len(values), 100):
                chunk = values[start : start + 100]
                batch = self._post(
                    "/predict/batch",
                    {"items": [{"text": value} for value in chunk]},
                )
                try:
                    results = batch["results"]
                    if len(results) != len(chunk):
                        raise TypeError
                    scores.extend(
                        self._probability(item["risk_score"], "risk_score")
                        for item in results
                    )
                except (KeyError, TypeError) as exc:
                    raise EnsembleUnavailableError(
                        "Remote batch response does not match its API contract."
                    ) from exc
            return scores

        return EnsembleComputation(
            ensemble=ensemble,
            members=members,
            score_batch=score_batch,
        )

    def close(self) -> None:
        self.client.close()
