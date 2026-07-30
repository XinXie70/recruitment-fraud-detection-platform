from __future__ import annotations

import importlib
import logging
import math
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Protocol, Sequence

import httpx

from config import settings

logger = logging.getLogger("fake_job_detection_api.models")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    module_path: str
    raw_batch_function: str = "_predict_risk_score"
    artifact_relative_path: str | None = None


@dataclass(frozen=True)
class RawModelResult:
    key: str
    display_name: str
    status: str
    score: float | None = None
    error: str | None = None
    error_code: str | None = None


class ModelArtifactUnavailableError(RuntimeError):
    """Raised before import when a required local model weight is unavailable."""


class RawProbabilityAdapter(Protocol):
    key: str
    display_name: str

    def predict_raw(self, text: str) -> float: ...

    def predict_raw_batch(self, texts: Sequence[str]) -> list[float]: ...


DEFAULT_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "logistic_regression",
        "Logistic Regression",
        "final_model_pipelines.lr_pipeline.predict",
    ),
    ModelSpec("svm", "SVM", "final_model_pipelines.svm_pipeline.predict"),
    ModelSpec("xgboost", "XGBoost", "final_model_pipelines.xgboost_pipeline.predict"),
    ModelSpec("dnn", "Deep Neural Network", "final_model_pipelines.dnn_pipeline.predict"),
    ModelSpec("rnn", "RNN", "final_model_pipelines.rnn_pipeline.predict"),
    ModelSpec("bilstm", "Bi-LSTM", "final_model_pipelines.bilstm_pipeline.predict"),
    ModelSpec(
        "bert",
        "BERT",
        "final_model_pipelines.bert_pipeline.predict",
        artifact_relative_path=(
            "model/final_model_pipelines/bert_pipeline/saved_model/model.safetensors"
        ),
    ),
    ModelSpec(
        "roberta",
        "RoBERTa",
        "final_model_pipelines.roberta_pipeline.predict",
        artifact_relative_path=(
            "model/final_model_pipelines/roberta_pipeline/saved_model/model.safetensors"
        ),
    ),
)


class ModelAdapter:
    """Stable wrapper around one existing model pipeline.

    Only this class knows the current pipelines expose a private
    ``_predict_risk_score`` batch function. Other application layers depend on
    ``predict_raw`` and ``predict_raw_batch`` instead.
    """

    def __init__(self, spec: ModelSpec, project_root: Path | None = None):
        self.spec = spec
        self.key = spec.key
        self.display_name = spec.display_name
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self._predict_batch: Callable[[list[str]], object] | None = None
        self._load_lock = Lock()

    def _check_required_artifact(self) -> None:
        relative_path = self.spec.artifact_relative_path
        if relative_path is None:
            return

        artifact = self.project_root / relative_path
        if not artifact.is_file():
            raise ModelArtifactUnavailableError(
                f"{self.display_name} model weight is missing."
            )

        with artifact.open("rb") as handle:
            prefix = handle.read(200)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ModelArtifactUnavailableError(
                f"{self.display_name} model weight is only a Git LFS pointer; "
                "retrieve the real artifact with Git LFS."
            )

    def _load(self) -> Callable[[list[str]], object]:
        if self._predict_batch is not None:
            return self._predict_batch

        with self._load_lock:
            if self._predict_batch is None:
                self._check_required_artifact()
                module = importlib.import_module(self.spec.module_path)
                function = getattr(module, self.spec.raw_batch_function, None)
                if function is None or not callable(function):
                    raise AttributeError(
                        f"{self.spec.module_path} does not expose callable "
                        f"{self.spec.raw_batch_function}"
                    )
                self._predict_batch = function
        return self._predict_batch

    @staticmethod
    def _normalise_scores(raw_scores: object, expected: int) -> list[float]:
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        scores = list(raw_scores)  # type: ignore[arg-type]
        if len(scores) != expected:
            raise ValueError(f"Expected {expected} scores but model returned {len(scores)}")

        normalised: list[float] = []
        for raw_score in scores:
            score = float(raw_score)
            if not math.isfinite(score) or score < 0 or score > 1:
                raise ValueError(f"Model returned invalid probability: {raw_score!r}")
            normalised.append(score)
        return normalised

    def predict_raw_batch(self, texts: Sequence[str]) -> list[float]:
        values = list(texts)
        if not values:
            return []
        raw_scores = self._load()(values)
        return self._normalise_scores(raw_scores, len(values))

    def predict_raw(self, text: str) -> float:
        return self.predict_raw_batch([text])[0]


#: Map from repository model keys to the deployed model server's endpoint keys.
_REMOTE_MODEL_KEY_MAP: dict[str, str] = {
    "logistic_regression": "lr",
    "svm": "svm",
    "xgboost": "xgboost",
    "dnn": "dnn",
    "rnn": "rnn",
    "bilstm": "bilstm",
    "bert": "bert",
    "roberta": "roberta",
}


class HttpModelAdapter:
    """Remote model adapter that calls the standalone Model Inference Server.

    Activated when ``MODEL_SERVER_URL`` is set. Each adapter sends requests to
    ``POST /predict/<key>`` with ``{"text": "..."}`` and reads the returned
    ``fraud_score`` probability. The current production model API exposes LR
    and BERT through this contract.
    """

    def __init__(self, key: str, display_name: str, base_url: str):
        self.key = key
        self.display_name = display_name
        self._base_url = base_url.rstrip("/")
        self._remote_key = _REMOTE_MODEL_KEY_MAP.get(key, key)
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=settings.model_server_timeout,
                    write=10.0,
                    pool=10.0,
                )
            )
        return self._client

    def predict_raw(self, text: str) -> float:
        try:
            resp = self.client.post(
                f"{self._base_url}/predict/{self._remote_key}",
                json={"text": text},
            )
            resp.raise_for_status()
            data = resp.json()
            if "fraud_score" not in data:
                raise RuntimeError(
                    f"Model {self.key} returned no fraud_score."
                )
            return ModelAdapter._normalise_scores([data["fraud_score"]], 1)[0]
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"HTTP error calling model server for {self.key}: {exc}"
            ) from exc

    def predict_raw_batch(self, texts: Sequence[str]) -> list[float]:
        return [self.predict_raw(text) for text in texts]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class ModelRegistry:
    def __init__(
        self,
        adapters: Sequence[RawProbabilityAdapter],
        max_workers: int | None = None,
    ):
        self.adapters = {adapter.key: adapter for adapter in adapters}
        self.max_workers = max_workers or settings.model_max_workers

    @classmethod
    def default(cls) -> "ModelRegistry":
        model_server_url = settings.model_server_url.strip()
        if model_server_url:
            logger.info(
                "Using remote model server at %s (MODEL_SERVER_URL is set)",
                model_server_url,
            )
            adapters: list[RawProbabilityAdapter] = [
                HttpModelAdapter(spec.key, spec.display_name, model_server_url)
                for spec in DEFAULT_MODEL_SPECS
            ]
        else:
            logger.info("Using local model pipelines (MODEL_SERVER_URL not set)")
            adapters = [ModelAdapter(spec) for spec in DEFAULT_MODEL_SPECS]
        return cls(adapters)

    def warm_up(self, sample: str, model_keys: Sequence[str]) -> dict[str, str | None]:
        outcomes: dict[str, str | None] = {}
        for key in model_keys:
            adapter = self.adapters.get(key)
            if adapter is None:
                outcomes[key] = "Model adapter is not registered."
                continue
            try:
                adapter.predict_raw(sample)
                outcomes[key] = None
            except Exception as exc:  # pragma: no cover - depends on model runtime
                outcomes[key] = f"{type(exc).__name__}: {exc}"
        return outcomes

    def predict_all(
        self,
        text: str,
        model_keys: Sequence[str],
        timeout_seconds: float,
    ) -> list[RawModelResult]:
        if not model_keys:
            return []

        worker_count = max(1, min(self.max_workers, len(model_keys)))
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="model")
        futures: dict[Future[float], RawProbabilityAdapter] = {}
        results_by_key: dict[str, RawModelResult] = {}

        for key in model_keys:
            adapter = self.adapters.get(key)
            if adapter is None:
                results_by_key[key] = RawModelResult(
                    key=key,
                    display_name=key,
                    status="error",
                    error="Model adapter is not registered.",
                    error_code="not_registered",
                )
                continue
            futures[executor.submit(adapter.predict_raw, text)] = adapter

        done, pending = wait(futures, timeout=timeout_seconds)
        for future in done:
            adapter = futures[future]
            try:
                results_by_key[adapter.key] = RawModelResult(
                    key=adapter.key,
                    display_name=adapter.display_name,
                    status="success",
                    score=future.result(),
                )
            except Exception as exc:
                logger.error(
                    "Model %s inference failed.",
                    adapter.key,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                artifact_error = isinstance(exc, ModelArtifactUnavailableError)
                results_by_key[adapter.key] = RawModelResult(
                    key=adapter.key,
                    display_name=adapter.display_name,
                    status="error",
                    error=(
                        str(exc)
                        if artifact_error
                        else f"{type(exc).__name__}: model inference failed."
                    ),
                    error_code=(
                        "artifact_unavailable" if artifact_error else "inference_failed"
                    ),
                )

        for future in pending:
            adapter = futures[future]
            future.cancel()
            results_by_key[adapter.key] = RawModelResult(
                key=adapter.key,
                display_name=adapter.display_name,
                status="timeout",
                error=f"Model exceeded {timeout_seconds:g} second timeout.",
                error_code="timeout",
            )

        executor.shutdown(wait=False, cancel_futures=True)
        return [results_by_key[key] for key in model_keys]

    def predict_raw_batches(
        self,
        texts: Sequence[str],
        model_keys: Sequence[str],
        timeout_seconds: float | None = None,
    ) -> dict[str, list[float]]:
        if not model_keys:
            return {}

        worker_count = max(1, min(self.max_workers, len(model_keys)))
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="xai-model")
        futures: dict[Future[list[float]], RawProbabilityAdapter] = {}
        for key in model_keys:
            adapter = self.adapters.get(key)
            if adapter is None:
                executor.shutdown(wait=False, cancel_futures=True)
                raise KeyError(f"Model adapter {key!r} is not registered")
            futures[executor.submit(adapter.predict_raw_batch, texts)] = adapter

        done, pending = wait(futures, timeout=timeout_seconds)
        if pending:
            timed_out = sorted(futures[future].key for future in pending)
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(
                f"Batch scoring exceeded {timeout_seconds:g} seconds for: "
                + ", ".join(timed_out)
            )

        outputs: dict[str, list[float]] = {}
        try:
            for future in done:
                adapter = futures[future]
                outputs[adapter.key] = future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return outputs
