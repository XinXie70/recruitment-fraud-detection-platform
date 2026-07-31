from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from config import settings

from final_model_pipelines.risk_mapping import apply_risk_mapping

from schemas.analysis import EnsembleResult, ModelMemberOutput
from services.model_adapter import ModelRegistry, RawModelResult


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


@dataclass(frozen=True)
class EnsembleConfig:
    version: str
    fitted: bool
    weight_source: str
    low_threshold: float
    high_threshold: float
    models: dict[str, EnsembleMemberConfig]

    @classmethod
    def load(cls, path: Path) -> "EnsembleConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        models: dict[str, EnsembleMemberConfig] = {}
        for key, model_payload in payload.get("models", {}).items():
            calibration_payload = model_payload.get("calibration", {"type": "identity"})
            calibration = CalibrationConfig(
                kind=str(calibration_payload.get("type", "identity")),
                coefficient=float(calibration_payload.get("coefficient", 1.0)),
                intercept=float(calibration_payload.get("intercept", 0.0)),
            )
            models[key] = EnsembleMemberConfig(
                weight=float(model_payload["weight"]),
                calibration=calibration,
            )

        config = cls(
            version=str(payload["version"]),
            fitted=bool(payload.get("fitted", False)),
            weight_source=str(payload.get("weight_source", "unknown")),
            low_threshold=float(payload["low_threshold"]),
            high_threshold=float(payload["high_threshold"]),
            models=models,
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
    """Resolve the default ensemble config path.
    """
    if env_path := settings.ensemble_config_path:
        return Path(env_path)


    # a "backend/" and "model/" folder (the project root).
    candidate = Path(__file__).resolve().parent
    for _ in range(6):  # safety limit — should never need more than 3-4 levels
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

    # Ultimate fallback
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
                )
            )

        mapping = apply_risk_mapping(
            risk_score,
            self.config.low_threshold,
            self.config.high_threshold,
        )
        risk_level = {
            "Likely Legitimate": "low",
            "Suspicious": "medium",
            "Likely Deceptive": "high",
        }[mapping["classification_label"]]
        failed_count = len(raw_results) - len(successful)
        status = "degraded" if failed_count else "success"
        ensemble = EnsembleResult(
            status=status,
            risk_score=risk_score,
            classification_label=mapping["classification_label"],
            risk_level=risk_level,
            prediction=mapping["prediction"],
            recommended_action=mapping["recommended_action"],
            low_threshold=self.config.low_threshold,
            high_threshold=self.config.high_threshold,
            active_model_count=len(successful),
            failed_model_count=failed_count,
            version=(
                f"{self.config.version}-degraded" if failed_count else self.config.version
            ),
            fitted=self.config.fitted,
            weight_source=self.config.weight_source,
        )

        active_keys = list(effective_weights)

        def score_batch(texts: Sequence[str]) -> list[float]:
            model_batches = self.registry.predict_raw_batches(
                texts,
                active_keys,
                timeout_seconds=self.timeout_seconds,
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
