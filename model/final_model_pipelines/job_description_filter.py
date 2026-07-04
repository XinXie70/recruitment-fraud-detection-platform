"""
Job description relevance filter.

This layer decides whether valid English text looks like a job posting. It does
not judge whether the posting is fake or real.
"""

from __future__ import annotations

import re
from typing import Any

MIN_JOB_RELEVANCE_SCORE = 0.4
WARNING_RELEVANCE_SCORE = 0.55

_JOB_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "title_role": ("job", "role", "position", "title", "opening", "vacancy", "opportunity"),
    "company": ("company", "employer", "organization", "organisation", "firm", "agency"),
    "responsibilities": ("responsibilities", "responsibility", "duties", "duty", "you will", "you'll"),
    "requirements": ("requirements", "requirement", "qualifications", "qualification", "must have", "required", "preferred"),
    "skills_experience": ("skills", "skill", "experience", "experienced", "proficient", "knowledge of"),
    "compensation": ("salary", "compensation", "benefits", "benefit", "package", "pay", "wage"),
    "employment_type": ("full-time", "full time", "part-time", "part time", "contract", "permanent", "temporary", "internship", "intern"),
    "location": ("location", "remote", "onsite", "on-site", "on site", "hybrid", "office", "based in"),
    "hiring_action": ("apply", "candidate", "hiring", "recruitment", "recruit", "interview", "resume", "résumé", " cv "),
}

_STRUCTURE_FIELD_KW: tuple[str, ...] = (
    "responsibilities",
    "requirements",
    "benefits",
    "qualifications",
    "duties",
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
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
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
    )
)

_NEGATIVE_TOPIC_KW: tuple[str, ...] = (
    "news",
    "article",
    "report",
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
    return sum(1 for keyword in keywords if keyword in lowered)


def _count_distinct_job_categories(text: str) -> int:
    return sum(1 for keywords in _JOB_CATEGORY_KEYWORDS.values() if _count_keyword_hits(text, keywords) > 0)


def _count_structure_fields(text: str) -> int:
    lowered = text.lower()
    return sum(1 for field in _STRUCTURE_FIELD_KW if field in lowered)


def _count_job_title_words(text: str) -> int:
    lowered = text.lower()
    return sum(1 for title in _JOB_TITLE_WORDS if re.search(rf"\b{re.escape(title)}\b", lowered))


def _count_hiring_action_words(text: str) -> int:
    lowered = f" {text.lower()} "
    return sum(1 for word in _HIRING_ACTION_WORDS if word in lowered)


def _count_job_phrases(text: str) -> int:
    return sum(1 for pattern in _JOB_PHRASE_PATTERNS if pattern.search(text))


def _count_negative_topics(text: str) -> int:
    lowered = text.lower()
    return sum(1 for topic in _NEGATIVE_TOPIC_KW if topic in lowered)


def compute_keyword_score(text: str) -> float:
    categories_hit = _count_distinct_job_categories(text)
    return round(min(categories_hit / len(_JOB_CATEGORY_KEYWORDS), 1.0), 4)


def compute_job_relevance_score(text: str) -> float:
    keyword_score = compute_keyword_score(text)
    structure_score = min(_count_structure_fields(text) / 2.0, 1.0)
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

    hiring_kw_hits = _count_keyword_hits(text, _JOB_CATEGORY_KEYWORDS["hiring_action"])
    has_role_signal = category_hits >= 1 and (
        _count_keyword_hits(text, _JOB_CATEGORY_KEYWORDS["title_role"]) > 0
        or title_hits > 0
        or _count_keyword_hits(text, _JOB_CATEGORY_KEYWORDS["employment_type"]) > 0
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
