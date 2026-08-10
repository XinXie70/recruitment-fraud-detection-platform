from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence, cast

from backend.config import settings

from final_model_pipelines.ensemble_pipeline.fp_gate import apply_risk
from final_model_pipelines.risk_mapping import apply_risk_mapping

from backend.schemas.analysis import EnsembleResult, ModelMemberOutput
from backend.services.model_adapter import ModelRegistry, RawModelResult


class EnsembleConfigurationError(ValueError):
    pass


class EnsembleUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class CalibrationConfig:
    kind: str
    coefficient: float = 1.0
    intercept: float = 0.0


@dataclass(frozen=True)
class EnsembleMemberConfig:
    weight: float
    calibration: CalibrationConfig
    role: Literal["weighted_member", "primary_score", "false_positive_gate"] = (
        "weighted_member"
    )


@dataclass(frozen=True)
class EnsembleConfig:
    version: str
    fitted: bool
    weight_source: str
    low_threshold: float
    high_threshold: float
    models: dict[str, EnsembleMemberConfig]
    method: Literal["calibrated_weighted", "bert_lr_fp_gate"] = "calibrated_weighted"
    bert_low_threshold: float | None = None
    bert_high_threshold: float | None = None
    lr_gate: float | None = None

    @classmethod
    def load(cls, path: Path) -> "EnsembleConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        method = cast(
            Literal["calibrated_weighted", "bert_lr_fp_gate"],
            str(payload.get("method", "calibrated_weighted")),
        )
        models: dict[str, EnsembleMemberConfig] = {}
        for key, model_payload in payload.get("models", {}).items():
            calibration_payload = model_payload.get("calibration", {"type": "identity"})
            calibration = CalibrationConfig(
                kind=str(calibration_payload.get("type", "identity")),
                coefficient=float(calibration_payload.get("coefficient", 1.0)),
                intercept=float(calibration_payload.get("intercept", 0.0)),
            )
            role_raw = str(model_payload.get("role", "weighted_member"))
            role = cast(
                Literal["weighted_member", "primary_score", "false_positive_gate"],
                role_raw
                if role_raw in {"weighted_member", "primary_score", "false_positive_gate"}
                else "weighted_member",
            )
            default_weight = 1.0 if method == "bert_lr_fp_gate" and key == "bert" else (
                0.0 if method == "bert_lr_fp_gate" else None
            )
            weight = float(
                model_payload["weight"]
                if "weight" in model_payload
                else default_weight
                if default_weight is not None
                else 0.0
            )
            models[key] = EnsembleMemberConfig(
                weight=weight,
                calibration=calibration,
                role=role,
            )

        low_threshold = float(
            payload.get(
                "bert_low_threshold",
                payload.get("low_threshold", 0.0),
            )
        )
        high_threshold = float(
            payload.get(
                "bert_high_threshold",
                payload.get("high_threshold", 1.0),
            )
        )
        config = cls(
            version=str(payload["version"]),
            fitted=bool(payload.get("fitted", False)),
            weight_source=str(payload.get("weight_source", "unknown")),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            models=models,
            method=method,
            bert_low_threshold=(
                float(payload["bert_low_threshold"])
                if "bert_low_threshold" in payload
                else low_threshold
            ),
            bert_high_threshold=(
                float(payload["bert_high_threshold"])
                if "bert_high_threshold" in payload
                else high_threshold
            ),
            lr_gate=float(payload["lr_gate"]) if "lr_gate" in payload else None,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.models:
            raise EnsembleConfigurationError("Ensemble must configure at least one model")
        if not 0 <= self.low_threshold < self.high_threshold <= 1:
            raise EnsembleConfigurationError("Ensemble thresholds are invalid")
        if any(member.weight < 0 for member in self.models.values()):
            raise EnsembleConfigurationError("Ensemble weights cannot be negative")

        if self.method == "bert_lr_fp_gate":
            required = {"logistic_regression", "bert"}
            if set(self.models) != required:
                raise EnsembleConfigurationError(
                    "FP-gate ensemble requires exactly logistic_regression and bert"
                )
            if self.lr_gate is None or not 0 <= self.lr_gate <= 1:
                raise EnsembleConfigurationError("FP-gate ensemble requires lr_gate in [0, 1]")
            if self.bert_low_threshold is None or self.bert_high_threshold is None:
                raise EnsembleConfigurationError(
                    "FP-gate ensemble requires bert_low_threshold and bert_high_threshold"
                )
            if not 0 <= self.bert_low_threshold < self.bert_high_threshold <= 1:
                raise EnsembleConfigurationError("FP-gate BERT thresholds are invalid")
        else:
            total_weight = sum(member.weight for member in self.models.values())
            if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
                raise EnsembleConfigurationError(
                    f"Configured ensemble weights must sum to 1, got {total_weight}"
                )

        supported_calibrators = {"identity", "sigmoid"}
        invalid = {
            member.calibration.kind
            for member in self.models.values()
            if member.calibration.kind not in supported_calibrators
        }
        if invalid:
            raise EnsembleConfigurationError(f"Unsupported calibrators: {sorted(invalid)}")


@dataclass(frozen=True)
class EnsembleComputation:
    ensemble: EnsembleResult
    members: list[ModelMemberOutput]
    score_batch: Callable[[Sequence[str]], list[float]]


def default_config_path() -> Path:
    """Resolve the default ensemble config path."""
    if env_path := settings.ensemble_config_path:
        return Path(env_path)

    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / "backend").is_dir() and (candidate / "model").is_dir():
            return (
                candidate
                / "model"
                / "final_model_pipelines"
                / "ensemble_pipeline"
                / "saved_model"
                / "ensemble_config.json"
            )
        candidate = candidate.parent

    import warnings

    fallback = Path(__file__).resolve().parents[2]
    warnings.warn(
        f"Could not auto-detect project root; using fallback {fallback}",
        stacklevel=2,
    )
    return (
        fallback
        / "model"
        / "final_model_pipelines"
        / "ensemble_pipeline"
        / "saved_model"
        / "ensemble_config.json"
    )


def apply_calibration(probability: float, config: CalibrationConfig) -> float:
    probability = min(1.0, max(0.0, float(probability)))
    if config.kind == "identity":
        return probability

    clipped = min(1 - 1e-7, max(1e-7, probability))
    logit = math.log(clipped / (1 - clipped))
    calibrated_logit = config.coefficient * logit + config.intercept
    if calibrated_logit >= 0:
        exp_value = math.exp(-calibrated_logit)
        return 1 / (1 + exp_value)
    exp_value = math.exp(calibrated_logit)
    return exp_value / (1 + exp_value)


_RISK_LEVEL_TO_API = {
    "Low": "low",
    "Suspicious": "medium",
    "High": "high",
}
_LABELS: dict[
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


class EnsemblePredictor:
    def __init__(
        self,
        registry: ModelRegistry,
        config: EnsembleConfig,
        timeout_seconds: float | None = None,
    ):
        self.registry = registry
        self.config = config
        self.config.validate()
        self.timeout_seconds = timeout_seconds or settings.model_timeout_seconds

    @classmethod
    def from_environment(cls, registry: ModelRegistry) -> "EnsemblePredictor":
        path = default_config_path()
        return cls(registry=registry, config=EnsembleConfig.load(path))

    @property
    def model_keys(self) -> list[str]:
        return list(self.config.models)

    def warm_up(self, sample: str) -> dict[str, str | None]:
        return self.registry.warm_up(sample, self.model_keys)

    def _successful_members(
        self, raw_results: Sequence[RawModelResult]
    ) -> tuple[list[RawModelResult], float]:
        successful = [
            item
            for item in raw_results
            if item.status == "success" and item.score is not None
        ]
        available_weight = sum(
            self.config.models[item.key].weight for item in successful
        )
        return successful, available_weight

    def predict(self, text: str) -> EnsembleComputation:
        if self.config.method == "bert_lr_fp_gate":
            return self._predict_fp_gate(text)
        return self._predict_weighted(text)

    def _predict_fp_gate(self, text: str) -> EnsembleComputation:
        raw_results = self.registry.predict_all(
            text,
            model_keys=["logistic_regression", "bert"],
            timeout_seconds=self.timeout_seconds,
        )
        by_key = {item.key: item for item in raw_results}
        lr = by_key.get("logistic_regression")
        bert = by_key.get("bert")
        if (
            lr is None
            or bert is None
            or lr.status != "success"
            or bert.status != "success"
            or lr.score is None
            or bert.score is None
        ):
            errors = "; ".join(
                f"{item.key}: {item.error or item.status}" for item in raw_results
            )
            raise EnsembleUnavailableError(f"No ensemble model succeeded. {errors}")

        assert self.config.bert_low_threshold is not None
        assert self.config.bert_high_threshold is not None
        assert self.config.lr_gate is not None

        risk = apply_risk(
            bert_score=float(bert.score),
            lr_score=float(lr.score),
            bert_high_threshold=self.config.bert_high_threshold,
            lr_gate=self.config.lr_gate,
            bert_low_threshold=self.config.bert_low_threshold,
        )
        risk_level = cast(
            Literal["low", "medium", "high"],
            _RISK_LEVEL_TO_API[str(risk["risk_level"])],
        )
        classification, prediction, action = _LABELS[risk_level]
        source = cast(Literal["bert", "lr_gate"], risk["risk_score_source"])
        members = [
            ModelMemberOutput(
                key="logistic_regression",
                display_name=lr.display_name,
                status="success",
                raw_score=lr.score,
                calibrated_score=lr.score,
                role="false_positive_gate",
                decision_active=source == "lr_gate",
            ),
            ModelMemberOutput(
                key="bert",
                display_name=bert.display_name,
                status="success",
                raw_score=bert.score,
                calibrated_score=bert.score,
                role="primary_score",
                decision_active=source == "bert",
            ),
        ]
        ensemble = EnsembleResult(
            status="success",
            risk_score=float(risk["risk_score"]),
            classification_label=classification,
            risk_level=risk_level,
            prediction=prediction,
            recommended_action=action,
            low_threshold=self.config.bert_low_threshold,
            high_threshold=self.config.bert_high_threshold,
            active_model_count=2,
            failed_model_count=0,
            version=self.config.version,
            fitted=self.config.fitted,
            weight_source=self.config.weight_source,
            method="bert_lr_fp_gate",
            risk_score_source=source,
            gate_triggered=bool(risk["gate_triggered"]),
            decision_reason=str(risk["decision_reason"]),
            bert_low_threshold=self.config.bert_low_threshold,
            bert_high_threshold=self.config.bert_high_threshold,
            lr_gate_threshold=self.config.lr_gate,
        )

        def score_batch(texts: Sequence[str]) -> list[float]:
            batches = self.registry.predict_raw_batches(
                texts,
                ["logistic_regression", "bert"],
                timeout_seconds=None,
            )
            outputs: list[float] = []
            for lr_score, bert_score in zip(
                batches["logistic_regression"], batches["bert"], strict=True
            ):
                item = apply_risk(
                    bert_score=float(bert_score),
                    lr_score=float(lr_score),
                    bert_high_threshold=self.config.bert_high_threshold or 0.3,
                    lr_gate=self.config.lr_gate or 0.06,
                    bert_low_threshold=self.config.bert_low_threshold or 0.0024,
                )
                outputs.append(float(item["risk_score"]))
            return outputs

        return EnsembleComputation(
            ensemble=ensemble,
            members=members,
            score_batch=score_batch,
        )

    def _predict_weighted(self, text: str) -> EnsembleComputation:
        raw_results = self.registry.predict_all(
            text,
            model_keys=self.model_keys,
            timeout_seconds=self.timeout_seconds,
        )
        successful, available_weight = self._successful_members(raw_results)
        if not successful or available_weight <= 0:
            errors = "; ".join(
                f"{item.key}: {item.error or item.status}" for item in raw_results
            )
            raise EnsembleUnavailableError(f"No ensemble model succeeded. {errors}")

        members: list[ModelMemberOutput] = []
        effective_weights: dict[str, float] = {}
        risk_score = 0.0

        for raw in raw_results:
            member_config = self.config.models[raw.key]
            if raw.status != "success" or raw.score is None:
                members.append(
                    ModelMemberOutput(
                        key=raw.key,
                        display_name=raw.display_name,
                        status=raw.status,
                        configured_weight=member_config.weight,
                        effective_weight=0,
                        weighted_contribution=0,
                        role=member_config.role,
                        error=raw.error,
                        error_code=raw.error_code,
                    )
                )
                continue

            calibrated_score = apply_calibration(raw.score, member_config.calibration)
            effective_weight = member_config.weight / available_weight
            contribution = calibrated_score * effective_weight
            effective_weights[raw.key] = effective_weight
            risk_score += contribution
            members.append(
                ModelMemberOutput(
                    key=raw.key,
                    display_name=raw.display_name,
                    status="success",
                    raw_score=raw.score,
                    calibrated_score=calibrated_score,
                    configured_weight=member_config.weight,
                    effective_weight=effective_weight,
                    weighted_contribution=contribution,
                    role=member_config.role,
                )
            )

        mapping = apply_risk_mapping(
            risk_score,
            self.config.low_threshold,
            self.config.high_threshold,
        )
        risk_level = cast(
            Literal["low", "medium", "high"],
            {
                "Likely Legitimate": "low",
                "Suspicious": "medium",
                "Likely Deceptive": "high",
            }[mapping["classification_label"]],
        )
        failed_count = len(raw_results) - len(successful)
        status: Literal["success", "degraded"] = (
            "degraded" if failed_count else "success"
        )
        ensemble = EnsembleResult(
            status=status,
            risk_score=risk_score,
            classification_label=cast(
                Literal["Likely Legitimate", "Suspicious", "Likely Deceptive"],
                mapping["classification_label"],
            ),
            risk_level=risk_level,
            prediction=cast(Literal["real", "fake"], mapping["prediction"]),
            recommended_action=cast(
                Literal["Safe", "Review Required", "High Risk Warning"],
                mapping["recommended_action"],
            ),
            low_threshold=self.config.low_threshold,
            high_threshold=self.config.high_threshold,
            active_model_count=len(successful),
            failed_model_count=failed_count,
            version=(
                f"{self.config.version}-degraded" if failed_count else self.config.version
            ),
            fitted=self.config.fitted,
            weight_source=self.config.weight_source,
            method="calibrated_weighted",
        )

        active_keys = list(effective_weights)

        def score_batch(texts: Sequence[str]) -> list[float]:
            model_batches = self.registry.predict_raw_batches(
                texts,
                active_keys,
                timeout_seconds=None,
            )
            outputs = [0.0] * len(texts)
            for key in active_keys:
                member_config = self.config.models[key]
                weight = effective_weights[key]
                for index, raw_score in enumerate(model_batches[key]):
                    outputs[index] += weight * apply_calibration(
                        raw_score, member_config.calibration
                    )
            return outputs

        return EnsembleComputation(
            ensemble=ensemble,
            members=members,
            score_batch=score_batch,
        )
