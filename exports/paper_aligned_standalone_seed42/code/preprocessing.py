"""Text cleaning and field concatenation for BERT inputs.

BERT keeps context signals (URLs, amounts, punctuation). Only light cleaning
is applied. Fixed splits already contain `combined_text`; this module still
supports tagged re-assembly when raw-style columns are present.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from config import (
    FIELD_TAG_MAP,
    PREFER_COMBINED_TEXT,
    TAGGED_FIELD_ORDER,
    TEXT_COLUMN,
    TEXT_FIELDS,
    USE_TAGGED_FORMAT,
)


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n\s*\n+")
_URL = re.compile(
    r"(https?://\S+|www\.\S+)",
    re.IGNORECASE,
)
_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}"
)


def missing_to_empty(value: Any) -> str:
    """Convert missing / literal 'nan' values to empty string."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def clean_text(
    value: Any,
    *,
    replace_url: bool = True,
    replace_email: bool = True,
    replace_phone: bool = False,
) -> str:
    """Light cleaning suitable for BERT (no stemming / stopword removal)."""
    text = missing_to_empty(value)
    if not text:
        return ""

    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub(" ", text)

    if replace_url:
        text = _URL.sub(" [URL] ", text)
    if replace_email:
        text = _EMAIL.sub(" [EMAIL] ", text)
    if replace_phone:
        text = _PHONE.sub(" [PHONE] ", text)

    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n", text)
    return text.strip()


def build_tagged_text(row: Dict[str, Any], field_order: Optional[Iterable[str]] = None) -> str:
    """Concatenate available fields with fixed tags, never dropping the row."""
    order = list(field_order) if field_order is not None else list(TAGGED_FIELD_ORDER)
    parts: List[str] = []
    for field in order:
        if field not in row:
            continue
        cleaned = clean_text(row.get(field, ""))
        if not cleaned:
            continue
        tag = FIELD_TAG_MAP.get(field, field.upper())
        parts.append(f"[{tag}] {cleaned}")
    return "\n".join(parts)


def build_tagged_from_combined(combined: str) -> str:
    """Best-effort section tags when only `combined_text` is available.

    Pipeline text is newline-joined from TEXT_FIELDS with empty fields dropped.
    Exact field recovery is impossible when fields contain newlines, so we map:
      - line 0 -> TITLE
      - line 1 -> COMPANY PROFILE (if present)
      - remaining lines split across DESCRIPTION / REQUIREMENTS / BENEFITS
    """
    text = clean_text(combined, replace_url=True, replace_email=True)
    if not text:
        return ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return f"[TITLE] {lines[0]}"

    parts: List[str] = [f"[TITLE] {lines[0]}"]
    rest = lines[1:]
    if len(rest) == 1:
        parts.append(f"[DESCRIPTION] {rest[0]}")
        return "\n".join(parts)

    parts.append(f"[COMPANY PROFILE] {rest[0]}")
    body = rest[1:]
    if not body:
        return "\n".join(parts)
    if len(body) == 1:
        parts.append(f"[DESCRIPTION] {body[0]}")
        return "\n".join(parts)

    # Split remaining lines into up to three body sections.
    n = len(body)
    cuts = [max(1, n // 3), max(2, (2 * n) // 3)]
    desc = "\n".join(body[: cuts[0]])
    reqs = "\n".join(body[cuts[0] : cuts[1]])
    bens = "\n".join(body[cuts[1] :])
    if desc:
        parts.append(f"[DESCRIPTION] {desc}")
    if reqs:
        parts.append(f"[REQUIREMENTS] {reqs}")
    if bens:
        parts.append(f"[BENEFITS] {bens}")
    return "\n".join(parts)


def build_plain_combined(row: Dict[str, Any], fields: Optional[Iterable[str]] = None) -> str:
    """Newline-join cleaned non-empty fields (matches shared data contract)."""
    order = list(fields) if fields is not None else list(TEXT_FIELDS)
    parts = [clean_text(row.get(field, "")) for field in order]
    return "\n".join(p for p in parts if p)


def extract_text_from_row(
    row: Dict[str, Any],
    *,
    prefer_combined: bool = PREFER_COMBINED_TEXT,
    use_tagged: bool = USE_TAGGED_FORMAT,
) -> str:
    """Resolve model input text for one advertisement row."""
    available = [f for f in TAGGED_FIELD_ORDER if f in row and missing_to_empty(row.get(f, ""))]

    # Tagged format takes priority when enabled.
    if use_tagged:
        if available:
            return build_tagged_text(row)
        if TEXT_COLUMN in row:
            tagged = build_tagged_from_combined(row.get(TEXT_COLUMN, ""))
            if tagged:
                return tagged

    if prefer_combined and TEXT_COLUMN in row:
        combined = clean_text(row.get(TEXT_COLUMN, ""), replace_url=True, replace_email=True)
        if combined:
            return combined

    if available:
        return build_plain_combined(row, available)

    if TEXT_COLUMN in row:
        combined = clean_text(row.get(TEXT_COLUMN, ""), replace_url=True, replace_email=True)
        if combined:
            return combined

    # Fallback: any string-like leftover columns except ids/labels.
    skip = {
        "label",
        "fraudulent",
        "record_id",
        "group_id",
        "in_balanced_dataset",
        "model_text",
        "split",
        "row_index",
    }
    leftovers = [
        clean_text(v)
        for k, v in row.items()
        if k not in skip and isinstance(v, (str, int, float))
    ]
    return "\n".join(x for x in leftovers if x)


def prepare_dataframe_text(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a standardised `model_text` column."""
    out = df.copy()
    texts = [
        extract_text_from_row(row._asdict() if hasattr(row, "_asdict") else row.to_dict())
        for _, row in out.iterrows()
    ]
    out["model_text"] = texts
    return out


def derive_title_preview(text: str, max_chars: int = 120) -> str:
    """Best-effort title from first line of combined text."""
    if not text:
        return ""
    first = text.split("\n", 1)[0].strip()
    return first[:max_chars]


def derive_company_preview(text: str, max_chars: int = 120) -> str:
    """Best-effort company snippet from second non-empty line."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return ""
    return lines[1][:max_chars]
