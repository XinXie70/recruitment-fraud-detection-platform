from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx

from backend.schemas.analysis import EnsembleResult, ModelMemberOutput


class EnsembleUnavailableError(RuntimeError):
    """Raised when the production FP-gate service cannot return a valid result."""


@dataclass(frozen=True)
class EnsembleComputation:
    ensemble: EnsembleResult
    members: list[ModelMemberOutput]
    score_batch: Callable[[Sequence[str]], list[float]]


_RISK_LEVELS = {"Low": "low", "Suspicious": "medium", "High": "high"}
_DISPLAY_NAMES = {"lr": "Logistic Regression", "bert": "BERT"}


class FPGatePredictor:
    """Client for the deployed BERT-primary + LR false-positive-gate API."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 120.0,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        headers = {}
        key = (api_key or "").strip()
        if key:
            headers["X-API-Key"] = key
        self.client = httpx.Client(timeout=timeout_seconds, headers=headers)

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = self.client.post(f"{self.base_url}{path}", json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EnsembleUnavailableError("FP-gate model service request failed.") from exc
        if not isinstance(data, dict):
            raise EnsembleUnavailableError("FP-gate model service returned an error.")
        if isinstance(data.get("ok"), bool) and data.get("ok") is False:
            raise EnsembleUnavailableError("FP-gate model service returned an error.")
        return data

    @staticmethod
    def _probability(value: str | int | float, field: str) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise EnsembleUnavailableError(
                f"FP-gate model service returned invalid {field}."
            ) from exc
        if not 0 <= score <= 1:
            raise EnsembleUnavailableError(
                f"FP-gate model service returned invalid {field}."
            )
        return score

    def warm_up(self, sample: str) -> dict[str, str | None]:
        try:
            self.predict(sample)
        except EnsembleUnavailableError as exc:
            return {"final_ensemble": str(exc)}
        return {"final_ensemble": None}

    def _predict_all(self, text: str) -> tuple[float, float, dict[str, Any]]:
        data = self._post("/predict/all", {"text": text})
        try:
            lr_score = self._probability(data["lr"]["lr_score"], "lr_score")
            bert_score = self._probability(
                data["bert"]["bert_score"], "bert_score"
            )
            risk = data["risk"]
        except (KeyError, TypeError) as exc:
            raise EnsembleUnavailableError(
                "FP-gate response does not match its API contract."
            ) from exc
        return lr_score, bert_score, risk

    def predict(self, text: str) -> EnsembleComputation:
        lr_score, bert_score, risk = self._predict_all(text)
        try:
            risk_score = self._probability(risk["risk_score"], "risk_score")
            risk_level = cast(
                Literal["low", "medium", "high"],
                _RISK_LEVELS[risk["risk_level"]],
            )
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
            raw_source = str(risk["risk_score_source"])
            if raw_source not in {"bert", "lr_gate"}:
                raise TypeError
            source = cast(Literal["bert", "lr_gate"], raw_source)
            gate_triggered = risk["gate_triggered"]
            if not isinstance(gate_triggered, bool):
                raise TypeError
            decision_reason = str(risk["decision_reason"])
        except (KeyError, TypeError) as exc:
            raise EnsembleUnavailableError(
                "FP-gate response does not match its API contract."
            ) from exc

        labels: dict[
            Literal["low", "medium", "high"],
            tuple[
                Literal["Likely Legitimate", "Suspicious", "Likely Deceptive"],
                Literal["real", "fake"],
                Literal["Safe", "Review Required", "High Risk Warning"],
            ],
        ] = {
            "low": ("Likely Legitimate", "real", "Safe"),
            "medium": ("Suspicious", "fake", "Review Required"),
            "high": ("Likely Deceptive", "fake", "High Risk Warning"),
        }
        classification, prediction, action = labels[risk_level]
        scores = {"lr": lr_score, "bert": bert_score}
        roles: dict[str, Literal["primary_score", "false_positive_gate"]] = {
            "lr": "false_positive_gate",
            "bert": "primary_score",
        }
        members = [
            ModelMemberOutput(
                key=key,
                display_name=_DISPLAY_NAMES[key],
                status="success",
                raw_score=score,
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
            version="bert-lr-fp-gate-v1",
            fitted=True,
            decision_strategy="bert_primary_lr_fp_gate",
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
                        "FP-gate batch response does not match its API contract."
                    ) from exc
            return scores

        return EnsembleComputation(
            ensemble=ensemble,
            members=members,
            score_batch=score_batch,
        )

    def close(self) -> None:
        self.client.close()
