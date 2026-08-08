from __future__ import annotations

import math
import logging
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from config import settings

from .contracts import EvidenceSpan, XAIResult

logger = logging.getLogger("fake_job_detection_api.xai")

BatchScorer = Callable[[Sequence[str]], list[float]]
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
COARSE_BOUNDARY_PATTERN = re.compile(r"[.!?]+(?=\s|$)|\n+", re.UNICODE)
INLINE_SEPARATOR_PATTERN = re.compile(r"[ \t/&+-]+")
CLAUSE_BOUNDARY_PATTERN = re.compile(r"[.!?;:\n]")
MIN_DISPLAY_ABS_CONTRIBUTION = 1e-12
RELATIVE_NOISE_FLOOR = 0.01
MIN_CONTEXT_WORDS = 3
MAX_CONTEXT_WORDS = 4
LONG_TEXT_PHRASE_WORDS = 8
LONG_TEXT_EVALS_PER_ITEM = 8

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
    full get gets getting got job jobs join now position positions recruit recruited recruiter
    recruiters recruiting recruitment required requirement requirements respond role roles
    service services small team teams time training trainings work working
    """.split()
)


@dataclass(frozen=True)
class TextSegment:
    start: int
    end: int


@dataclass(frozen=True)
class PartitionAttribution:
    segments: list[TextSegment]
    contributions: list[float]
    base_value: float | None


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


def _trim_segment(text: str, start: int, end: int) -> TextSegment | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end or WORD_PATTERN.search(text, start, end) is None:
        return None
    return TextSegment(start, end)


def _split_segment_by_word_limit(
    text: str,
    segment: TextSegment,
    max_words: int,
) -> list[TextSegment]:
    words = list(WORD_PATTERN.finditer(text, segment.start, segment.end))
    if len(words) <= max_words:
        return [segment]

    chunks: list[TextSegment] = []
    for index in range(0, len(words), max_words):
        group = words[index : index + max_words]
        chunks.append(TextSegment(group[0].start(), group[-1].end()))
    return chunks


def _build_coarse_segments(text: str, max_words: int) -> list[TextSegment]:
    """Split long input into sentence or line spans with bounded word counts."""

    segments: list[TextSegment] = []
    cursor = 0
    for boundary in COARSE_BOUNDARY_PATTERN.finditer(text):
        segment = _trim_segment(text, cursor, boundary.end())
        if segment is not None:
            segments.extend(_split_segment_by_word_limit(text, segment, max_words))
        cursor = boundary.end()

    tail = _trim_segment(text, cursor, len(text))
    if tail is not None:
        segments.extend(_split_segment_by_word_limit(text, tail, max_words))

    if segments:
        return segments

    whole_text = _trim_segment(text, 0, len(text))
    if whole_text is None:
        return []
    return _split_segment_by_word_limit(text, whole_text, max_words)


def _build_phrase_segments(text: str, max_words: int) -> list[TextSegment]:
    """Build non-overlapping phrase spans without crossing sentence boundaries."""

    return _build_coarse_segments(text, max_words=max_words)


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


def _is_numeric_word(word: str) -> bool:
    return word.replace(",", "").replace(".", "").isdigit()


def _is_noise_fragment(item: EvidenceSpan) -> bool:
    words = _evidence_words(item)
    meaningful = [
        word
        for word in words
        if word not in FUNCTION_WORDS
        and word not in CALENDAR_WORDS
        and word not in GENERIC_RECRUITING_WORDS
        and not _is_numeric_word(word)
    ]
    if not meaningful:
        return True
    return len(meaningful) == 1 and len(meaningful[0]) < 4


def _is_context_worthy_noise(item: EvidenceSpan) -> bool:
    """Allow generic model signals to become phrases, but reject pure dates/numbers."""

    excluded = FUNCTION_WORDS | CALENDAR_WORDS
    return any(
        word in GENERIC_RECRUITING_WORDS
        or (word not in excluded and not _is_numeric_word(word))
        for word in _evidence_words(item)
    )


def _same_clause(text: str, left: EvidenceSpan, right: EvidenceSpan) -> bool:
    start = min(left.end, right.end)
    end = max(left.start, right.start)
    return CLAUSE_BOUNDARY_PATTERN.search(text[start:end]) is None


def _display_threshold(items: list[EvidenceSpan]) -> float:
    """Keep relatively important evidence even when the model output is tiny."""

    largest = max((abs(item.contribution) for item in items), default=0.0)
    return max(MIN_DISPLAY_ABS_CONTRIBUTION, largest * RELATIVE_NOISE_FLOOR)


def _context_window(
    text: str,
    item: EvidenceSpan,
    max_words: int,
) -> dict[str, int]:
    """Expand an isolated attribution into a readable phrase in the same clause."""

    clause_start = item.start
    while clause_start > 0 and text[clause_start - 1] not in ".!?;:\n":
        clause_start -= 1
    clause_end = item.end
    while clause_end < len(text) and text[clause_end] not in ".!?;:\n":
        clause_end += 1

    words = list(WORD_PATTERN.finditer(text, clause_start, clause_end))
    if not words:
        return {"start": item.start, "end": item.end}

    overlapping = [
        index
        for index, word in enumerate(words)
        if word.start() < item.end and word.end() > item.start
    ]
    anchor_start = overlapping[0] if overlapping else 0
    anchor_end = overlapping[-1] if overlapping else anchor_start
    window_start = anchor_start
    window_end = anchor_end
    target_words = min(
        max_words,
        max(MIN_CONTEXT_WORDS, anchor_end - anchor_start + 1),
    )
    while window_end - window_start + 1 < target_words:
        can_expand_left = window_start > 0
        can_expand_right = window_end + 1 < len(words)
        if not can_expand_left and not can_expand_right:
            break
        if can_expand_left:
            window_start -= 1
        if window_end - window_start + 1 >= max_words:
            break
        if can_expand_right:
            window_end += 1

    return {
        "start": words[window_start].start(),
        "end": words[window_end].end(),
    }


def _limit_interval_words(
    text: str,
    interval: dict[str, int],
    items: list[EvidenceSpan],
    max_words: int,
) -> dict[str, int]:
    """Keep an overlapping context group centred on its strongest SHAP item."""

    words = list(WORD_PATTERN.finditer(text, interval["start"], interval["end"]))
    if len(words) <= max_words:
        return interval

    contained = [
        item
        for item in items
        if item.start >= interval["start"] and item.end <= interval["end"]
    ]
    anchor = max(contained, key=lambda item: abs(item.contribution), default=None)
    if anchor is None:
        selected_start = 0
    else:
        overlapping = [
            index
            for index, word in enumerate(words)
            if word.start() < anchor.end and word.end() > anchor.start
        ]
        anchor_index = overlapping[0] if overlapping else 0
        selected_start = max(0, anchor_index - max_words // 2)
        selected_start = min(selected_start, len(words) - max_words)

    selected = words[selected_start : selected_start + max_words]
    return {"start": selected[0].start(), "end": selected[-1].end()}


def _contextualize_noisy_items(
    items: list[EvidenceSpan],
    text: str,
    max_words: int = MAX_CONTEXT_WORDS,
) -> list[EvidenceSpan]:
    """Attach isolated high-impact fragments to nearby meaningful evidence."""

    threshold = _display_threshold(items)
    eligible = [item for item in items if abs(item.contribution) >= threshold]
    specific = [item for item in eligible if not _is_noise_fragment(item)]
    noisy = [item for item in eligible if _is_noise_fragment(item)]
    if not specific:
        groups = [
            _context_window(text, item, max_words)
            for item in sorted(
                noisy,
                key=lambda value: abs(value.contribution),
                reverse=True,
            )
        ]
    else:
        groups = [
            {"start": item.start, "end": item.end}
            for item in sorted(specific, key=lambda value: value.start)
        ]

    unassigned_noise: list[EvidenceSpan] = []
    for noise in (
        sorted(noisy, key=lambda value: abs(value.contribution), reverse=True)
        if specific
        else []
    ):
        choices: list[tuple[int, int, int, int]] = []
        for index, group in enumerate(groups):
            anchor = EvidenceSpan(
                text=text[group["start"] : group["end"]],
                start=group["start"],
                end=group["end"],
                contribution=1.0,
                direction="raises_risk",
            )
            if not _same_clause(text, noise, anchor):
                continue
            new_start = min(group["start"], noise.start)
            new_end = max(group["end"], noise.end)
            word_count = len(WORD_PATTERN.findall(text[new_start:new_end]))
            if word_count > max_words:
                continue
            gap_start = min(noise.end, anchor.end)
            gap_end = max(noise.start, anchor.start)
            distance = len(WORD_PATTERN.findall(text[gap_start:gap_end]))
            choices.append((distance, word_count, index, new_start))

        if not choices:
            unassigned_noise.append(noise)
            continue
        _, _, selected_index, new_start = min(choices)
        groups[selected_index]["start"] = new_start
        groups[selected_index]["end"] = max(groups[selected_index]["end"], noise.end)

    core_groups = list(groups)
    for noise in unassigned_noise:
        if not _is_context_worthy_noise(noise):
            continue
        window = _context_window(text, noise, max_words)
        for core in core_groups:
            if noise.start >= core["end"] and window["start"] < core["end"]:
                window["start"] = core["end"]
            elif noise.end <= core["start"] and window["end"] > core["start"]:
                window["end"] = core["start"]
        trimmed = _trim_segment(text, window["start"], window["end"])
        if trimmed is None:
            continue
        if len(WORD_PATTERN.findall(text[trimmed.start : trimmed.end])) < 2:
            continue
        groups.append({"start": trimmed.start, "end": trimmed.end})

    intervals: list[dict[str, int]] = []
    for group in sorted(groups, key=lambda value: value["start"]):
        if intervals and group["start"] < intervals[-1]["end"]:
            intervals[-1]["end"] = max(intervals[-1]["end"], group["end"])
        else:
            intervals.append(dict(group))

    contextualized: list[EvidenceSpan] = []
    for interval in intervals:
        interval = _limit_interval_words(text, interval, items, max_words)
        if interval["start"] > 0 and text[interval["start"] - 1] == "$":
            interval = {"start": interval["start"] - 1, "end": interval["end"]}
        contribution = sum(
            item.contribution
            for item in items
            if item.start >= interval["start"] and item.end <= interval["end"]
        )
        if abs(contribution) < threshold:
            continue
        contextualized.append(
            EvidenceSpan(
                text=text[interval["start"] : interval["end"]],
                start=interval["start"],
                end=interval["end"],
                contribution=contribution,
                direction=(
                    "raises_risk" if contribution > 0 else "lowers_risk"
                ),
            )
        )
    return contextualized


def _top_evidence(
    items: list[EvidenceSpan],
    limit: int,
    *,
    text: str,
    merge_shap_tokens: bool,
) -> list[EvidenceSpan]:
    candidates = [item for item in items if _contains_content(item, allow_generic=True)]
    if merge_shap_tokens:
        merged_candidates = _merge_adjacent_shap_items(candidates, text)
        candidates = _contextualize_noisy_items(merged_candidates, text)
        if not candidates:
            threshold = _display_threshold(merged_candidates)
            candidates = [
                item
                for item in merged_candidates
                if abs(item.contribution) >= threshold
                and _contains_content(item, allow_generic=True)
            ]
    else:
        candidates = [
            item
            for item in candidates
            if abs(item.contribution) >= MIN_DISPLAY_ABS_CONTRIBUTION
            and not _is_noise_fragment(item)
        ]
    selected = sorted(
        candidates,
        key=lambda item: abs(item.contribution),
        reverse=True,
    )[:limit]
    return sorted(selected, key=lambda item: item.start)


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

        def tokenizer(candidate: str, return_offsets_mapping: bool = True) -> dict:
            segments = segment_builder(candidate)
            if not segments and candidate:
                segments = [TextSegment(0, len(candidate))]
            output = {
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
