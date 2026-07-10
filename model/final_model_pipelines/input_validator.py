"""
Basic Input Validation — runs before job-description relevance filtering and model inference.

Checks whether user input is syntactically valid text (type, length, format).
Does NOT judge whether the content is a job posting; see job_description_filter.py.
"""

from __future__ import annotations

import re
from typing import Any

MIN_ENGLISH_WORDS = 8
_MIN_LETTER_RATIO = 0.35

_CODE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in (
        r"```",
        r"^\s*def\s+\w+\s*\(",
        r"^\s*function\s+\w+\s*\(",
        r"^\s*import\s+\w+",
        r"^\s*class\s+\w+.*:",
        r"console\.log\s*\(",
        r"public\s+static\s+void\s+main",
        r"#include\s*<",
        r"^\s*var\s+\w+\s*=",
        r"^\s*let\s+\w+\s*=",
        r"^\s*const\s+\w+\s*=",
        r"\{\s*\n\s*\"[\w-]+\"\s*:",
    )
)

_CHAT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^(?:hi|hello|hey|thanks|thank you|lol|haha|ok|yes|no|bye)[\s!.?]*$",
        r"^(?:how are you|what's up|good morning|good night)[\s!.?]*$",
        r"^(?:asdf|qwerty|test test|lorem ipsum)[\s!.?]*$",
    )
)

_URL_ONLY_PATTERN = re.compile(
    r"^(?:https?://|www\.)[^\s]+(?:\s+(?:https?://|www\.)[^\s]+)*$",
    re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"[a-zA-Z]+")


def _count_english_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def _letter_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0
    letters = sum(1 for ch in compact if ch.isalpha())
    return letters / len(compact)


def _is_only_url(text: str) -> bool:
    return bool(_URL_ONLY_PATTERN.match(text.strip()))


def _looks_like_gibberish(text: str) -> bool:
    if _letter_ratio(text) < _MIN_LETTER_RATIO:
        return True

    words = _WORD_PATTERN.findall(text)
    if not words:
        return True

    short_weird = sum(
        1 for w in words if len(w) <= 3 and not re.search(r"[aeiouAEIOU]", w)
    )
    return short_weird / len(words) > 0.6


def _looks_like_code(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CODE_PATTERNS)


def _looks_like_chat_or_noise(text: str) -> bool:
    stripped = text.strip()
    if any(pattern.match(stripped) for pattern in _CHAT_PATTERNS):
        return True

    words = _WORD_PATTERN.findall(stripped)
    if len(words) <= 12:
        casual_markers = ("lol", "haha", "btw", "omg", "idk", "tbh", "brb")
        lowered = stripped.lower()
        if any(marker in lowered for marker in casual_markers):
            return True
    return False


def validate_input_text(text: str) -> dict[str, Any]:
    """
    Basic text validity check (layer 1).

    Returns:
        {
            "is_valid": bool,
            "status": "valid" / "invalid_input",
            "reason": str,
        }
    """
    if not isinstance(text, str):
        return {
            "is_valid": False,
            "status": "invalid_input",
            "reason": "Input must be a string.",
        }

    stripped = text.strip()
    if not stripped:
        return {
            "is_valid": False,
            "status": "invalid_input",
            "reason": "Input cannot be empty.",
        }

    if _is_only_url(stripped):
        return {
            "is_valid": False,
            "status": "invalid_input",
            "reason": "Input cannot be only a URL.",
        }

    if _looks_like_gibberish(stripped):
        return {
            "is_valid": False,
            "status": "invalid_input",
            "reason": "Input appears to be mostly numbers, symbols, or unreadable text.",
        }

    if _looks_like_code(stripped):
        return {
            "is_valid": False,
            "status": "invalid_input",
            "reason": "Input appears to be a code snippet rather than a job posting.",
        }

    if _looks_like_chat_or_noise(stripped):
        return {
            "is_valid": False,
            "status": "invalid_input",
            "reason": "Input appears to be casual chat or meaningless text.",
        }

    word_count = _count_english_words(stripped)
    if word_count < MIN_ENGLISH_WORDS:
        return {
            "is_valid": False,
            "status": "invalid_input",
            "reason": (
                f"Input is too short ({word_count} English words; "
                f"at least {MIN_ENGLISH_WORDS} required)."
            ),
        }

    return {
        "is_valid": True,
        "status": "valid",
        "reason": "",
    }
