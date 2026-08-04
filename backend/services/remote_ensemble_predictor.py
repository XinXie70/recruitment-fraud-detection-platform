from __future__ import annotations

from collections.abc import Sequence

import httpx

from schemas.analysis import EnsembleResult, ModelMemberOutput
from services.ensemble_predictor import EnsembleComputation, EnsembleUnavailableError


_RISK_LEVELS = {"Low": "low", "Suspicious": "medium", "High": "high"}
_DISPLAY_NAMES = {"lr": "Logistic Regression", "bert": "BERT"}


class RemoteFinalEnsemblePredictor:
    """Adapter for the deployed LR + BERT FP-gate model API."""

    def __init__(self, base_url: str, timeout_seconds: float = 120.0):
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
        if not isinstance(data, dict) or data.get("ok") is not True:
            raise EnsembleUnavailableError("Remote final ensemble returned an error.")
        return data

    @staticmethod
    def _probability(value: object, field: str) -> float:
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

    def predict(self, text: str) -> EnsembleComputation:
        data = self._post("/predict/all", {"text": text})
        try:
            lr_score = self._probability(data["lr"]["lr_score"], "lr_score")
            bert_score = self._probability(
                data["bert"]["bert_score"], "bert_score"
            )
            risk = data["risk"]
            risk_score = self._probability(risk["risk_score"], "risk_score")
            risk_level = _RISK_LEVELS[risk["risk_level"]]
            thresholds = risk["thresholds"]
            low_threshold = self._probability(
                thresholds["bert_low_threshold"], "bert_low_threshold"
            )
            high_threshold = self._probability(
                thresholds["bert_high_threshold"], "bert_high_threshold"
            )
            source = str(risk["risk_score_source"])
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
        members = [
            ModelMemberOutput(
                key=key,
                display_name=_DISPLAY_NAMES[key],
                status="success",
                raw_score=score,
                calibrated_score=score,
                configured_weight=1.0 if source.startswith(key) else 0.0,
                effective_weight=1.0 if source.startswith(key) else 0.0,
                weighted_contribution=risk_score if source.startswith(key) else 0.0,
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
