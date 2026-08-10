"""Lightweight text normalisation shared by LR and BERT adapters."""

from __future__ import annotations

import re


_WHITESPACE = re.compile(r"\s+")


def prepare_text_from_input(text: str) -> str:
    cleaned = str(text or "").replace("\x00", " ").strip()
    return _WHITESPACE.sub(" ", cleaned)
