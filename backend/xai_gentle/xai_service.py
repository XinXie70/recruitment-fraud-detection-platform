from __future__ import annotations

import logging
from typing import Callable, Sequence

from backend.config import settings

from .contracts import EvidenceSpan, XAIResult
from .evidence import (
    BatchScorer, LONG_TEXT_EVALS_PER_ITEM, LONG_TEXT_PHRASE_WORDS,
    MIN_DISPLAY_ABS_CONTRIBUTION, PartitionAttribution, TextSegment,
    TokenizerOutput, WORD_PATTERN, _build_coarse_segments, _build_phrase_segments,
    _build_segments, _top_evidence,
)

logger = logging.getLogger("fake_job_detection_api.xai")

class XAIService:
    """Explain the exact FP-gate risk scorer with word or long-text phrase SHAP."""

    def __init__(
        self,
        max_items: int | None = None,
        max_evals: int | None = None,
        hierarchical_min_words: int | None = None,
        hierarchical_top_segments: int | None = None,
        hierarchical_segment_words: int | None = None,
        hierarchical_max_evals: int | None = None,
    ):
        self.max_items = settings.xai_max_items if max_items is None else max_items
        self.max_evals = settings.xai_max_evals if max_evals is None else max_evals
        self.hierarchical_min_words = (
            settings.xai_hierarchical_min_words
            if hierarchical_min_words is None
            else hierarchical_min_words
        )
        self.hierarchical_top_segments = (
            settings.xai_hierarchical_top_segments
            if hierarchical_top_segments is None
            else hierarchical_top_segments
        )
        self.hierarchical_segment_words = (
            settings.xai_hierarchical_segment_words
            if hierarchical_segment_words is None
            else hierarchical_segment_words
        )
        self.hierarchical_max_evals = (
            settings.xai_hierarchical_max_evals
            if hierarchical_max_evals is None
            else hierarchical_max_evals
        )

        numeric_settings = {
            "max_items": self.max_items,
            "max_evals": self.max_evals,
            "hierarchical_min_words": self.hierarchical_min_words,
            "hierarchical_top_segments": self.hierarchical_top_segments,
            "hierarchical_segment_words": self.hierarchical_segment_words,
            "hierarchical_max_evals": self.hierarchical_max_evals,
        }
        invalid = [name for name, value in numeric_settings.items() if value <= 0]
        if invalid:
            raise ValueError(
                "XAI numeric settings must be positive: " + ", ".join(invalid)
            )

    def explain(
        self,
        text: str,
        score_batch: BatchScorer,
        expected_output: float,
    ) -> XAIResult:
        try:
            word_count = len(WORD_PATTERN.findall(text))
            if word_count >= self.hierarchical_min_words:
                return self._explain_hierarchical_shap(
                    text,
                    score_batch,
                    expected_output,
                )
            return self._explain_word_level_shap(text, score_batch, expected_output)
        except Exception:
            logger.warning("SHAP explanation failed", exc_info=True)
            return XAIResult(
                status="unavailable",
                method="unavailable",
                output_value=expected_output,
                message="SHAP explanation is temporarily unavailable.",
            )

    def _partition_attributions(
        self,
        value: str,
        score_batch: BatchScorer,
        segment_builder: Callable[[str], list[TextSegment]],
        max_evals: int,
    ) -> PartitionAttribution:
        import numpy as np
        import shap

        original_segments = segment_builder(value)
        if not original_segments:
            raise ValueError("No explainable text spans were found in the input")

        def tokenizer(
            candidate: str, return_offsets_mapping: bool = True
        ) -> TokenizerOutput:
            segments = segment_builder(candidate)
            if not segments and candidate:
                segments = [TextSegment(0, len(candidate))]
            output: TokenizerOutput = {
                "input_ids": [candidate[item.start : item.end] for item in segments]
            }
            if return_offsets_mapping:
                output["offset_mapping"] = [(item.start, item.end) for item in segments]
            return output

        def model_function(values: Sequence[str]):
            probabilities = score_batch([str(value) for value in values])
            return np.asarray([[1 - score, score] for score in probabilities])

        masker = shap.maskers.Text(tokenizer, mask_token="...", collapse_mask_token=True)
        explainer = shap.Explainer(
            model_function,
            masker,
            algorithm="partition",
            output_names=["real", "fake"],
        )
        explanation = explainer([value], max_evals=max_evals)
        values = np.asarray(explanation.values)
        if values.ndim != 3 or values.shape[2] < 2:
            raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
        contributions = values[0, :, 1]
        if len(contributions) != len(original_segments):
            raise ValueError("SHAP token count does not match original text offsets")

        base_values = np.asarray(explanation.base_values)
        base_value = float(base_values[0, 1]) if base_values.ndim == 2 else None
        return PartitionAttribution(
            segments=original_segments,
            contributions=[float(item) for item in contributions],
            base_value=base_value,
        )

    def _explain_word_level_shap(
        self,
        text: str,
        score_batch: BatchScorer,
        expected_output: float,
    ) -> XAIResult:
        attribution = self._partition_attributions(
            text,
            score_batch,
            lambda value: _build_segments(value, max_segments=100000),
            self.max_evals,
        )
        items: list[EvidenceSpan] = []
        for segment, contribution in zip(
            attribution.segments,
            attribution.contributions,
            strict=True,
        ):
            if abs(contribution) < MIN_DISPLAY_ABS_CONTRIBUTION:
                continue
            items.append(
                EvidenceSpan(
                    text=text[segment.start : segment.end],
                    start=segment.start,
                    end=segment.end,
                    contribution=contribution,
                    direction=(
                        "raises_risk" if contribution > 0 else "lowers_risk"
                    ),
                )
            )

        return XAIResult(
            status="success",
            method="shap_partition",
            base_value=attribution.base_value,
            output_value=expected_output,
            items=_top_evidence(
                items,
                self.max_items,
                text=text,
                merge_shap_tokens=True,
            ),
        )

    def _explain_hierarchical_shap(
        self,
        text: str,
        score_batch: BatchScorer,
        expected_output: float,
    ) -> XAIResult:
        # Long advertisements are explained as sentence-aware phrase blocks across
        # the complete input. The previous two-stage implementation first selected
        # long paragraphs and then re-masked only those paragraphs. For saturated
        # probabilities, the second pass could collapse every fine attribution to
        # zero even though the first pass found non-zero evidence. A single phrase
        # pass keeps every attribution tied to the formal scorer, returns useful
        # original-text offsets, and avoids repeating the most expensive SHAP stage.
        total_budget = min(
            self.max_evals,
            self.hierarchical_max_evals,
            max(8, self.max_items * LONG_TEXT_EVALS_PER_ITEM),
        )
        attribution = self._partition_attributions(
            text,
            score_batch,
            lambda value: _build_phrase_segments(
                value,
                max_words=LONG_TEXT_PHRASE_WORDS,
            ),
            total_budget,
        )
        phrase_items: list[EvidenceSpan] = []
        for segment, contribution in zip(
            attribution.segments,
            attribution.contributions,
            strict=True,
        ):
            if abs(contribution) < MIN_DISPLAY_ABS_CONTRIBUTION:
                continue
            phrase_items.append(
                EvidenceSpan(
                    text=text[segment.start : segment.end],
                    start=segment.start,
                    end=segment.end,
                    contribution=contribution,
                    direction=(
                        "raises_risk" if contribution > 0 else "lowers_risk"
                    ),
                )
            )

        evidence = _top_evidence(
            phrase_items,
            self.max_items,
            text=text,
            merge_shap_tokens=False,
        )
        if evidence:
            detail_message = (
                "Long-text Partition SHAP evaluated sentence-aware phrase spans "
                "across the complete advertisement."
            )
        else:
            detail_message = (
                "Partition SHAP found no reliable phrase-level evidence to display."
            )

        return XAIResult(
            status="success",
            method="shap_partition",
            base_value=attribution.base_value,
            output_value=expected_output,
            items=evidence,
            message=detail_message,
        )
