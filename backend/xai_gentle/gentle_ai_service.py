from __future__ import annotations

import json
import logging
from pathlib import Path

from config import settings
from pydantic import BaseModel, Field

from .contracts import (
    EducationItem,
    GentleAIResult,
    GentleEvidenceExplanation,
    RiskContext,
    XAIResult,
)
from .gentle_fallback import build_template_guidance

logger = logging.getLogger("fake_job_detection_api.gentle_ai")

class OllamaRewrite(BaseModel):
    summary: str = Field(..., min_length=1)
    evidence_explanations: list[str]


def default_knowledge_path() -> Path:
    return Path(__file__).resolve().parent / "knowledge" / "education_en.json"


class GentleAIService:
    """Turns structured ensemble/XAI data into cautious educational language."""

    def __init__(
        self,
        knowledge_path: Path | None = None,
        ollama_enabled: bool | None = None,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        ollama_timeout: float | None = None,
    ):
        path = knowledge_path or default_knowledge_path()
        raw_items = json.loads(path.read_text(encoding="utf-8"))
        self.items = [EducationItem.model_validate(item) for item in raw_items]
        self.ollama_enabled = (
            settings.ollama_enabled if ollama_enabled is None else ollama_enabled
        )
        self.ollama_base_url = (
            ollama_base_url or settings.ollama_base_url
        ).rstrip("/")
        self.ollama_model = ollama_model or settings.ollama_model
        self.ollama_timeout = ollama_timeout or settings.ollama_timeout

    def list_items(self, topic: str | None = None) -> list[EducationItem]:
        if topic is None:
            return list(self.items)
        return [item for item in self.items if item.topic == topic]

    def get_item(self, item_id: str) -> EducationItem | None:
        return next((item for item in self.items if item.id == item_id), None)

    def _select_learning_items(self, xai: XAIResult) -> list[EducationItem]:
        general = self.get_item("fake-job-general-checks")
        selected = [general] if general else []
        # Select education by topic relevance without changing the XAI direction.
        evidence_text = " ".join(item.text.lower() for item in xai.items)
        for item in self.items:
            if item.topic != "fake_jobs" or item in selected:
                continue
            if any(term.lower() in evidence_text for term in item.indicator_terms):
                selected.append(item)
        return selected[:3]

    def generate(self, risk: RiskContext, xai: XAIResult) -> GentleAIResult:
        learning_items = self._select_learning_items(xai)
        template = build_template_guidance(risk, xai, learning_items)
        if not self.ollama_enabled:
            template.message = "Local template guidance used; Ollama is disabled."
            return template
        if not self.ollama_model:
            template.message = "Local template guidance used; OLLAMA_MODEL is not configured."
            return template

        try:
            return self._rewrite_with_ollama(template)
        except Exception:
            logger.warning("Ollama rewrite failed", exc_info=True)
            template.message = "Local template guidance used because Ollama was unavailable."
            return template

    def _rewrite_with_ollama(self, template: GentleAIResult) -> GentleAIResult:
        import httpx

        with httpx.Client(timeout=self.ollama_timeout) as client:
            models_response = client.get(f"{self.ollama_base_url}/api/tags")
            models_response.raise_for_status()
            model_names = {
                model.get("name") or model.get("model")
                for model in models_response.json().get("models", [])
            }
            if self.ollama_model not in model_names:
                raise RuntimeError(f"Configured model {self.ollama_model!r} is not installed")

            source = {
                "summary": template.summary,
                "evidence_explanations": [
                    item.explanation for item in template.evidence_explanations
                ],
            }
            prompt = (
                "Rewrite the supplied English text in calm, clear language for a job "
                "seeker. Preserve every factual claim and uncertainty. Do not add new "
                "reasons, decisions, facts, or advice. Return only the requested JSON.\n"
                + json.dumps(source, ensure_ascii=True)
            )
            response = client.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": OllamaRewrite.model_json_schema(),
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            rewrite = OllamaRewrite.model_validate_json(content)

        if len(rewrite.evidence_explanations) != len(template.evidence_explanations):
            raise ValueError("Ollama changed the number of evidence explanations")
        evidence = [
            GentleEvidenceExplanation(
                text=item.text,
                start=item.start,
                end=item.end,
                direction=item.direction,
                explanation=rewrite.evidence_explanations[index],
            )
            for index, item in enumerate(template.evidence_explanations)
        ]
        return GentleAIResult(
            status="success",
            provider="ollama",
            summary=rewrite.summary,
            evidence_explanations=evidence,
            next_steps=template.next_steps,
            learning_item_ids=template.learning_item_ids,
            disclaimer=template.disclaimer,
        )
