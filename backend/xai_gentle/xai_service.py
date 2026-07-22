from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from config import settings

from .contracts import EvidenceSpan, XAIResult


BatchScorer = Callable[[Sequence[str]], list[float]]
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
INLINE_SEPARATOR_PATTERN = re.compile(r"[ \t/&+-]+")

FUNCTION_WORDS = frozenset(
    """
    a about after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few
    for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just me more most my myself no nor not of off on once
    only or other our ours ourselves out over own same she should so some such than that
    the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves via
    """.split()
)
CALENDAR_WORDS = frozenset(
    """
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october november december
    """.split()
)
GENERIC_RECRUITING_WORDS = frozenset(
    """
    applicant applicants application applications apply applying candidate candidates
    ask asked asking asks client clients company companies detail details employee employees
    employer employers employment
    full job jobs join position positions recruit recruited recruiter recruiters recruiting
    recruitment required requirement requirements respond role roles service services
    team teams time work working
    """.split()
)


@dataclass(frozen=True)
class TextSegment:
    start: int
    end: int


def _build_segments(text: str, max_segments: int) -> list[TextSegment]:
    matches = list(WORD_PATTERN.finditer(text))
    if not matches:
        return []
    if len(matches) <= max_segments:
        return [TextSegment(match.start(), match.end()) for match in matches]

    group_size = math.ceil(len(matches) / max_segments)
    segments: list[TextSegment] = []
    for index in range(0, len(matches), group_size):
        group = matches[index : index + group_size]
        segments.append(TextSegment(group[0].start(), group[-1].end()))
    return segments


def _evidence_words(item: EvidenceSpan) -> list[str]:
    return [match.group(0).casefold() for match in WORD_PATTERN.finditer(item.text)]


def _contains_content(item: EvidenceSpan, *, allow_generic: bool) -> bool:
    excluded = FUNCTION_WORDS | CALENDAR_WORDS
    if not allow_generic:
        excluded = excluded | GENERIC_RECRUITING_WORDS
    return any(word not in excluded for word in _evidence_words(item))


def _merge_adjacent_shap_items(
    items: list[EvidenceSpan],
    text: str,
    max_words: int = 3,
) -> list[EvidenceSpan]:
    """Combine additive, same-direction SHAP tokens into readable short phrases."""

    merged: list[EvidenceSpan] = []
    for item in sorted(items, key=lambda value: value.start):
        if not merged:
            merged.append(item)
            continue

        previous = merged[-1]
        separator = text[previous.end : item.start]
        combined_word_count = len(_evidence_words(previous)) + len(_evidence_words(item))
        can_merge = (
            previous.direction == item.direction
            and item.start >= previous.end
            and bool(separator)
            and INLINE_SEPARATOR_PATTERN.fullmatch(separator) is not None
            and combined_word_count <= max_words
        )
        if not can_merge:
            merged.append(item)
            continue

        merged[-1] = EvidenceSpan(
            text=text[previous.start : item.end],
            start=previous.start,
            end=item.end,
            contribution=previous.contribution + item.contribution,
            direction=previous.direction,
        )
    return merged


def _top_evidence(
    items: list[EvidenceSpan],
    limit: int,
    *,
    text: str,
    merge_shap_tokens: bool,
) -> list[EvidenceSpan]:
    candidates = [item for item in items if _contains_content(item, allow_generic=True)]
    if merge_shap_tokens:
        candidates = _merge_adjacent_shap_items(candidates, text)
    candidates = [item for item in candidates if _contains_content(item, allow_generic=False)]

    selected = sorted(
        candidates,
        key=lambda item: abs(item.contribution),
        reverse=True,
    )[:limit]
    return sorted(selected, key=lambda item: item.start)


class XAIService:
    """Explains the exact request-specific ensemble scorer.

    SHAP PartitionExplainer is preferred. When SHAP is unavailable, the service
    uses leave-one-span-out occlusion against the same scorer. The fallback is
    still model-derived attribution; it is not the frontend keyword rule list.
    """

    def __init__(
        self,
        prefer_shap: bool | None = None,
        max_items: int | None = None,
        max_segments: int | None = None,
        max_evals: int | None = None,
    ):
        self.prefer_shap = prefer_shap if prefer_shap is not None else settings.xai_use_shap
        self.max_items = max_items or settings.xai_max_items
        self.max_segments = max_segments or settings.xai_max_segments
        self.max_evals = max_evals or settings.xai_max_evals

    def explain(
        self,
        text: str,
        score_batch: BatchScorer,
        expected_output: float,
    ) -> XAIResult:
        shap_error: str | None = None
        if self.prefer_shap:
            try:
                return self._explain_with_shap(text, score_batch, expected_output)
            except Exception as exc:  # SHAP is optional at runtime
                shap_error = f"{type(exc).__name__}: {exc}"

        try:
            result = self._explain_with_occlusion(text, score_batch, expected_output)
            if shap_error:
                result.message = (
                    "SHAP was unavailable, so ensemble occlusion attribution was used. "
                    f"Reason: {shap_error}"
                )
            return result
        except Exception as exc:
            messages = [f"Occlusion explanation failed: {type(exc).__name__}: {exc}"]
            if shap_error:
                messages.insert(0, f"SHAP explanation failed: {shap_error}")
            return XAIResult(
                status="unavailable",
                method="unavailable",
                output_value=expected_output,
                message=" ".join(messages),
            )

    def _explain_with_occlusion(
        self,
        text: str,
        score_batch: BatchScorer,
        expected_output: float,
    ) -> XAIResult:
        segments = _build_segments(text, self.max_segments)
        if not segments:
            return XAIResult(
                status="success",
                method="occlusion_fallback",
                output_value=expected_output,
                items=[],
                message="No explainable word spans were found in the input.",
            )

        masked_texts = [
            f"{text[:segment.start]}{' ' * (segment.end - segment.start)}{text[segment.end:]}"
            for segment in segments
        ]
        baseline_and_masks = score_batch([text, *masked_texts])
        if len(baseline_and_masks) != len(masked_texts) + 1:
            raise ValueError("Ensemble scorer returned an unexpected batch length")
        baseline = float(baseline_and_masks[0])

        items: list[EvidenceSpan] = []
        for segment, masked_score in zip(segments, baseline_and_masks[1:]):
            contribution = baseline - float(masked_score)
            if abs(contribution) < 1e-6:
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

        message = None
        if not math.isclose(baseline, expected_output, abs_tol=1e-5):
            message = (
                "The explanation scorer output differed slightly from the initial "
                "ensemble output."
            )
        return XAIResult(
            status="success",
            method="occlusion_fallback",
            base_value=None,
            output_value=baseline,
            items=_top_evidence(
                items,
                self.max_items,
                text=text,
                merge_shap_tokens=False,
            ),
            message=message,
        )

    def _explain_with_shap(
        self,
        text: str,
        score_batch: BatchScorer,
        expected_output: float,
    ) -> XAIResult:
        import numpy as np
        import shap

        original_segments = _build_segments(text, max_segments=100000)
        if not original_segments:
            raise ValueError("No explainable word spans were found in the input")

        def tokenizer(value: str, return_offsets_mapping: bool = True) -> dict:
            segments = _build_segments(value, max_segments=100000)
            if not segments and value:
                segments = [TextSegment(0, len(value))]
            output = {"input_ids": [value[item.start : item.end] for item in segments]}
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
        explanation = explainer([text], max_evals=self.max_evals)
        values = np.asarray(explanation.values)
        if values.ndim != 3 or values.shape[2] < 2:
            raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
        contributions = values[0, :, 1]
        if len(contributions) != len(original_segments):
            raise ValueError("SHAP token count does not match original text offsets")

        items: list[EvidenceSpan] = []
        for segment, contribution_value in zip(original_segments, contributions):
            contribution = float(contribution_value)
            if abs(contribution) < 1e-6:
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

        base_values = np.asarray(explanation.base_values)
        base_value = float(base_values[0, 1]) if base_values.ndim == 2 else None
        return XAIResult(
            status="success",
            method="shap_partition",
            base_value=base_value,
            output_value=expected_output,
            items=_top_evidence(
                items,
                self.max_items,
                text=text,
                merge_shap_tokens=True,
            ),
        )
