"""
Job Description Relevance Filter — decides whether text looks like a job posting.

This module is independent from basic input validation (input_validator.py).
It does NOT judge whether a posting is fake or real; only whether the topic is job-related.
"""

from __future__ import annotations

import re
from typing import Any

MIN_JOB_RELEVANCE_SCORE = 0.4
WARNING_RELEVANCE_SCORE = 0.55

_JOB_TITLE_ROLE_KW: tuple[str, ...] = (
    "job",
    "role",
    "position",
    "title",
    "opening",
    "vacancy",
    "opportunity",
)
_JOB_COMPANY_KW: tuple[str, ...] = (
    "company",
    "employer",
    "organization",
    "organisation",
    "firm",
    "agency",
)
_JOB_RESPONSIBILITIES_KW: tuple[str, ...] = (
    "responsibilities",
    "responsibility",
    "duties",
    "duty",
    "you will",
    "you'll",
)
_JOB_REQUIREMENTS_KW: tuple[str, ...] = (
    "requirements",
    "requirement",
    "qualifications",
    "qualification",
    "must have",
    "required",
    "preferred",
)
_JOB_SKILLS_KW: tuple[str, ...] = (
    "skills",
    "skill",
    "experience",
    "experienced",
    "proficient",
    "knowledge of",
)
_JOB_COMPENSATION_KW: tuple[str, ...] = (
    "salary",
    "compensation",
    "benefits",
    "benefit",
    "package",
    "pay",
    "wage",
)
_JOB_EMPLOYMENT_TYPE_KW: tuple[str, ...] = (
    "full-time",
    "full time",
    "part-time",
    "part time",
    "contract",
    "permanent",
    "temporary",
    "internship",
    "intern",
)
_JOB_LOCATION_KW: tuple[str, ...] = (
    "location",
    "remote",
    "onsite",
    "on-site",
    "on site",
    "hybrid",
    "office",
    "based in",
)
_JOB_HIRING_ACTION_KW: tuple[str, ...] = (
    "apply",
    "candidate",
    "hiring",
    "recruitment",
    "recruit",
    "interview",
    "resume",
    "résumé",
    " cv ",
)

_JOB_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "title_role": _JOB_TITLE_ROLE_KW,
    "company": _JOB_COMPANY_KW,
    "responsibilities": _JOB_RESPONSIBILITIES_KW,
    "requirements": _JOB_REQUIREMENTS_KW,
    "skills_experience": _JOB_SKILLS_KW,
    "compensation": _JOB_COMPENSATION_KW,
    "employment_type": _JOB_EMPLOYMENT_TYPE_KW,
    "location": _JOB_LOCATION_KW,
    "hiring_action": _JOB_HIRING_ACTION_KW,
}

_STRUCTURE_FIELD_KW: tuple[str, ...] = (
    "responsibilities",
    "requirements",
    "benefits",
    "qualifications",
    "duties",
    "tasks",
    "successful candidate",
)

_JOB_TITLE_WORDS: tuple[str, ...] = (
    "engineer",
    "developer",
    "manager",
    "assistant",
    "analyst",
    "consultant",
    "intern",
    "officer",
    "specialist",
    "sales",
    "marketing",
    "teacher",
    "nurse",
    "designer",
    "administrator",
    "coordinator",
    "director",
    "executive",
    "technician",
    "accountant",
    "clerk",
    "representative",
    "supervisor",
    "architect",
    "scientist",
    "researcher",
)

_HIRING_ACTION_WORDS: tuple[str, ...] = (
    "hiring",
    "apply",
    "candidate",
    "recruitment",
    "recruit",
    "interview",
    "resume",
    "full-time",
    "full time",
    "part-time",
    "part time",
    "remote",
)

_JOB_PHRASE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwe(?:'re| are) looking for\b",
        r"\bthe ideal candidate\b",
        r"\bresponsibilities include\b",
        r"\brequirements include\b",
        r"\byou will be responsible\b",
        r"\bjoin our team\b",
        r"\babout (?:the |this )?role\b",
        r"\bjob description\b",
        r"\bposition summary\b",
        r"\bwhat you(?:'ll| will) do\b",
        r"\bwho you are\b",
        r"\bthe successful candidate\b",
    )
)

_NEGATIVE_TOPIC_KW: tuple[str, ...] = (
    "news",
    "article",
    "news report",
    "government",
    "election",
    "war",
    "movie",
    "film",
    "recipe",
    "travel",
    "hotel",
    "restaurant",
    "product review",
    "research paper",
    "abstract",
    "introduction",
    "methodology",
    "conclusion",
    "tutorial",
    "ingredients",
    "tablespoon",
    "protagonist",
    "screenplay",
    "box office",
    "itinerary",
    "sightseeing",
    "political",
    "parliament",
    "essay",
    "thesis",
    "dissertation",
    "bibliography",
    "figure 1",
    "figure 2",
    "api documentation",
    "documentation for",
    "release notes",
    "changelog",
)

_WORD_PATTERN = re.compile(r"[a-zA-Z]+")

_NOT_JOB_RELATED_REASON = (
    "Input text is valid English text but does not appear to be a job posting "
    "or job description."
)
_WARNING_REASON = (
    "Input may be a job posting, but it lacks common job description fields. "
    "Prediction may be less reliable."
)


def _count_english_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    lowered = f" {text.lower()} "
    return sum(1 for kw in keywords if kw in lowered)


def _count_category_hits(text: str) -> dict[str, int]:
    return {
        category: _count_keyword_hits(text, keywords)
        for category, keywords in _JOB_CATEGORY_KEYWORDS.items()
    }


def _count_distinct_job_categories(text: str) -> int:
    return sum(1 for hits in _count_category_hits(text).values() if hits > 0)


def _count_structure_fields(text: str) -> int:
    lowered = text.lower()
    return sum(1 for field in _STRUCTURE_FIELD_KW if field in lowered)


def _count_job_title_words(text: str) -> int:
    lowered = text.lower()
    return sum(
        1
        for title in _JOB_TITLE_WORDS
        if re.search(rf"\b{re.escape(title)}\b", lowered)
    )


def _count_hiring_action_words(text: str) -> int:
    lowered = f" {text.lower()} "
    return sum(1 for word in _HIRING_ACTION_WORDS if word in lowered)


def _count_job_phrases(text: str) -> int:
    return sum(1 for pattern in _JOB_PHRASE_PATTERNS if pattern.search(text))


def _count_negative_topics(text: str) -> int:
    lowered = text.lower()
    return sum(
        1
        for topic in _NEGATIVE_TOPIC_KW
        if re.search(rf"\b{re.escape(topic)}\b", lowered)
    )


def compute_keyword_score(text: str) -> float:
    """Proportion of job element categories present in the text."""
    categories_hit = _count_distinct_job_categories(text)
    return round(min(categories_hit / len(_JOB_CATEGORY_KEYWORDS), 1.0), 4)


def compute_job_relevance_score(text: str) -> float:
    """
    Composite job-relevance score in [0, 1].

    Combines keyword coverage, structure fields, title words, hiring actions,
    job-description phrasing, and penalties for non-job topics.
    """
    keyword_score = compute_keyword_score(text)
    structure_hits = _count_structure_fields(text)
    structure_score = min(structure_hits / 2.0, 1.0)

    title_hits = _count_job_title_words(text)
    title_score = min(title_hits / 2.0, 1.0)

    action_hits = _count_hiring_action_words(text)
    action_score = min(action_hits / 3.0, 1.0)

    phrase_hits = _count_job_phrases(text)
    phrase_score = min(phrase_hits / 2.0, 1.0)

    negative_hits = _count_negative_topics(text)
    negative_penalty = min(negative_hits * 0.08, 0.45)

    word_count = _count_english_words(text)
    category_hits = _count_distinct_job_categories(text)
    if word_count >= 50 and category_hits <= 1 and negative_hits >= 2:
        negative_penalty = min(negative_penalty + 0.25, 0.55)
    if word_count >= 80 and category_hits <= 2 and action_hits == 0 and phrase_hits == 0:
        negative_penalty = min(negative_penalty + 0.15, 0.55)

    raw_score = (
        0.30 * keyword_score
        + 0.20 * structure_score
        + 0.15 * title_score
        + 0.15 * action_score
        + 0.20 * phrase_score
    )

    hiring_kw_hits = _count_keyword_hits(text, _JOB_HIRING_ACTION_KW)
    has_role_signal = category_hits >= 1 and (
        _count_keyword_hits(text, _JOB_TITLE_ROLE_KW) > 0
        or title_hits > 0
        or _count_keyword_hits(text, _JOB_EMPLOYMENT_TYPE_KW) > 0
    )
    has_hiring_signal = action_hits >= 1 or hiring_kw_hits > 0
    if has_role_signal and has_hiring_signal and (title_hits >= 1 or action_hits >= 2):
        raw_score = max(raw_score, 0.42)
    elif hiring_kw_hits >= 2 and category_hits >= 2:
        raw_score = max(raw_score, 0.42)
    elif category_hits >= 3 and (action_hits >= 1 or phrase_hits >= 1):
        raw_score = max(raw_score, 0.41)

    return round(max(0.0, min(1.0, raw_score - negative_penalty)), 4)


def check_job_description_relevance(text: str) -> dict[str, Any]:
    """
    Determine whether text is job-posting / job-description related.

    Expects text that has already passed basic input validation.

    Returns:
        {
            "is_job_related": bool,
            "status": "valid" / "not_job_related" / "success_with_warning",
            "reason": str,
            "job_relevance_score": float,
        }
    """
    stripped = text.strip()
    job_relevance_score = compute_job_relevance_score(stripped)
    structure_hits = _count_structure_fields(stripped)

    if job_relevance_score < MIN_JOB_RELEVANCE_SCORE:
        return {
            "is_job_related": False,
            "status": "not_job_related",
            "reason": _NOT_JOB_RELATED_REASON,
            "job_relevance_score": job_relevance_score,
        }

    if job_relevance_score < WARNING_RELEVANCE_SCORE or structure_hits < 2:
        return {
            "is_job_related": True,
            "status": "success_with_warning",
            "reason": _WARNING_REASON,
            "job_relevance_score": job_relevance_score,
        }

    return {
        "is_job_related": True,
        "status": "valid",
        "reason": "",
        "job_relevance_score": job_relevance_score,
    }
