"""
Job Description Relevance Filter - lightweight layer-2 gate.

This module is independent from basic input validation (input_validator.py).
It does NOT judge whether a posting is fake or real. It only blocks text when it
is clearly non-recruitment content and has no job/recruitment signal. Ambiguous
or job-related text should pass through to the fake-job classifier.

Layer-2 keyword and regex rules are loaded from:
  input validation--data/job_relevance_keywords.csv
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

_WORD_PATTERN = re.compile(r"[a-zA-Z]+")
_OPENING_WORDS = 28
_KEYWORD_CSV_NAME = "job_relevance_keywords.csv"

_NOT_JOB_RELATED_REASON = (
    "Input text is valid English text, matches an obvious non-recruitment "
    "topic, and contains no job or recruitment signal."
)


def _keyword_csv_path() -> Path:
    root = Path(__file__).resolve().parent
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("input validation")
    ]
    if not candidates:
        raise FileNotFoundError("Could not find input validation data directory.")
    path = candidates[0] / _KEYWORD_CSV_NAME
    if not path.exists():
        raise FileNotFoundError(f"Could not find layer-2 keyword CSV: {path}")
    return path


def _load_keyword_rules() -> dict[str, Any]:
    terms: dict[str, list[str]] = {
        "role": [],
        "broad_role": [],
        "hiring_action": [],
    }
    detail_terms: dict[str, list[str]] = {}
    regex_patterns: dict[str, list[re.Pattern[str]]] = {
        "job_phrase": [],
        "news_or_article": [],
        "forum_or_discussion": [],
        "technical_or_general": [],
    }

    with _keyword_csv_path().open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("layer") != "layer_2":
                continue

            category = (row.get("category") or "").strip()
            match_type = (row.get("match_type") or "").strip()
            value = (row.get("keyword_or_pattern") or "").strip()
            polarity = (row.get("polarity") or "").strip()
            if not category or not match_type or not value:
                continue

            if match_type == "regex":
                flags = re.IGNORECASE
                if category == "forum_or_discussion":
                    flags |= re.MULTILINE
                regex_patterns.setdefault(category, []).append(re.compile(value, flags))
                continue

            if match_type != "term":
                continue

            if polarity == "positive" and category in terms:
                terms[category].append(value)
            elif polarity == "positive":
                detail_terms.setdefault(category, []).append(value)

    return {
        "role_terms": tuple(terms["role"]),
        "broad_role_terms": tuple(terms["broad_role"]),
        "hiring_action_terms": tuple(terms["hiring_action"]),
        "detail_terms": {key: tuple(values) for key, values in detail_terms.items()},
        "job_phrase_patterns": tuple(regex_patterns["job_phrase"]),
        "news_or_article_patterns": tuple(regex_patterns["news_or_article"]),
        "forum_or_discussion_patterns": tuple(regex_patterns["forum_or_discussion"]),
        "technical_or_general_patterns": tuple(regex_patterns["technical_or_general"]),
    }


_RULES = _load_keyword_rules()
_ROLE_TERMS: tuple[str, ...] = _RULES["role_terms"]
_BROAD_ROLE_TERMS: tuple[str, ...] = _RULES["broad_role_terms"]
_JOB_ACTION_TERMS: tuple[str, ...] = _RULES["hiring_action_terms"]
_DETAIL_CATEGORIES: dict[str, tuple[str, ...]] = _RULES["detail_terms"]
_JOB_PHRASE_PATTERNS: tuple[re.Pattern[str], ...] = _RULES["job_phrase_patterns"]
_NEWS_OR_ARTICLE_PATTERNS: tuple[re.Pattern[str], ...] = _RULES[
    "news_or_article_patterns"
]
_FORUM_OR_DISCUSSION_PATTERNS: tuple[re.Pattern[str], ...] = _RULES[
    "forum_or_discussion_patterns"
]
_OTHER_NON_JOB_PATTERNS: tuple[re.Pattern[str], ...] = _RULES[
    "technical_or_general_patterns"
]


def _count_english_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def _contains_term(text: str, term: str) -> bool:
    if " " in term or term.startswith("."):
        return term in text
    return bool(re.search(rf"\b{re.escape(term)}s?\b", text))


def _count_terms(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if _contains_term(lowered, term.lower()))


def _opening_text(text: str) -> str:
    return " ".join(_WORD_PATTERN.findall(text.lower())[:_OPENING_WORDS])


def _has_opening_role_signal(text: str) -> bool:
    opening = _opening_text(text)
    if not opening:
        return False
    if re.search(r"\b(?:job|jobs|position|role)\b", opening) and _count_terms(
        opening, _ROLE_TERMS + _BROAD_ROLE_TERMS
    ):
        return True
    return _count_terms(opening, _ROLE_TERMS) > 0


def _count_job_phrases(text: str) -> int:
    return sum(1 for pattern in _JOB_PHRASE_PATTERNS if pattern.search(text))


def _count_detail_categories(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        category: _count_terms(lowered, terms)
        for category, terms in _DETAIL_CATEGORIES.items()
    }


def _count_distinct_detail_categories(text: str) -> int:
    return sum(1 for hits in _count_detail_categories(text).values() if hits > 0)


def _has_money_signal(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"(?:\$|£|€)\s?\d|\b(?:usd|aud|cad|gbp)\b", lowered))


def _count_context_patterns(text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def _is_non_job_context(text: str) -> bool:
    return (
        _count_context_patterns(text, _NEWS_OR_ARTICLE_PATTERNS)
        + _count_context_patterns(text, _FORUM_OR_DISCUSSION_PATTERNS)
        + _count_context_patterns(text, _OTHER_NON_JOB_PATTERNS)
    ) > 0


def _has_any_job_signal(
    *,
    text: str,
    role_hits: int,
    action_hits: int,
    phrase_hits: int,
    detail_hits: dict[str, int],
    opening_role_signal: bool,
    non_job_context: bool,
) -> bool:
    """Return True when text has any job/recruitment signal worth passing on."""
    employment_signal = bool(
        re.search(
            r"\b(?:full[- ]time|part[- ]time|internship|permanent|temporary)\b",
            text.lower(),
        )
    )
    strong_detail_signal = any(
        detail_hits.get(category, 0) > 0
        for category in (
            "responsibilities",
            "requirements",
            "employment_type",
            "compensation",
            "company",
            "location",
        )
    )
    protected_job_signal = (
        action_hits > 0
        or phrase_hits > 0
        or opening_role_signal
        or employment_signal
        or detail_hits.get("compensation", 0) > 0
        or _has_money_signal(text)
    )
    if non_job_context:
        return protected_job_signal
    return (
        role_hits > 0
        or action_hits > 0
        or phrase_hits > 0
        or strong_detail_signal
        or opening_role_signal
        or _has_money_signal(text)
    )


def explain_job_relevance(text: str) -> dict[str, Any]:
    """
    Explain the job-relevance decision.

    The layer is a conservative gate: it fails only when obvious non-recruitment
    context is present and no job/recruitment signal is present.
    """
    stripped = text.strip()
    role_hits = _count_terms(stripped, _ROLE_TERMS)
    broad_role_hits = _count_terms(stripped, _BROAD_ROLE_TERMS)
    action_hits = _count_terms(stripped, _JOB_ACTION_TERMS)
    phrase_hits = _count_job_phrases(stripped)
    detail_hits = _count_detail_categories(stripped)
    detail_categories = sum(1 for hits in detail_hits.values() if hits > 0)
    opening_role_signal = _has_opening_role_signal(stripped)
    news_context_hits = _count_context_patterns(stripped, _NEWS_OR_ARTICLE_PATTERNS)
    forum_context_hits = _count_context_patterns(stripped, _FORUM_OR_DISCUSSION_PATTERNS)
    other_non_job_hits = _count_context_patterns(stripped, _OTHER_NON_JOB_PATTERNS)
    non_job_context = news_context_hits + forum_context_hits + other_non_job_hits > 0
    has_job_signal = _has_any_job_signal(
        text=stripped,
        role_hits=role_hits,
        action_hits=action_hits,
        phrase_hits=phrase_hits,
        detail_hits=detail_hits,
        opening_role_signal=opening_role_signal,
        non_job_context=non_job_context,
    )
    return {
        "role_hits": role_hits,
        "broad_role_hits": broad_role_hits,
        "action_hits": action_hits,
        "phrase_hits": phrase_hits,
        "detail_hits": detail_hits,
        "detail_categories": detail_categories,
        "opening_role_signal": opening_role_signal,
        "news_context_hits": news_context_hits,
        "forum_context_hits": forum_context_hits,
        "other_non_job_hits": other_non_job_hits,
        "non_job_context": non_job_context,
        "has_job_signal": has_job_signal,
        "final_gate_rule": (
            "fail only when non_job_context is true and has_job_signal is false; "
            "otherwise success"
        ),
        "word_count": _count_english_words(stripped),
    }


def check_job_description_relevance(text: str) -> dict[str, Any]:
    """
    Determine whether text is job-posting / job-description related.

    Expects text that has already passed basic input validation.
    """
    stripped = text.strip()
    explanation = explain_job_relevance(stripped)
    should_fail = explanation["non_job_context"] and not explanation["has_job_signal"]

    if should_fail:
        return {
            "is_job_related": False,
            "status": "fail",
            "reason": _NOT_JOB_RELATED_REASON,
            "relevance_explanation": explanation,
        }

    return {
        "is_job_related": True,
        "status": "success",
        "reason": "",
        "relevance_explanation": explanation,
    }
