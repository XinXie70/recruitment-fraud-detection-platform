#Normalize API request payloads into LR and BERT text inputs
from __future__ import annotations
from typing import Any, TypedDict
from settings import MAX_TEXT_CHARS, TEXT_FIELDS
class ResolvedTexts(TypedDict):
    record_id: str | None
    combined_text: str
    model_text: str
def _as_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text
def _enforce_text_length(text: str, *, field: str = "text") -> None:
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"{field} exceeds maximum length of {MAX_TEXT_CHARS} characters"
        )
def build_texts(payload: dict[str, Any]) -> ResolvedTexts:
    """Return plain combined_text (LR) and tagged model_text (BERT)."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    record_id = _as_str(payload.get("record_id")) or None
    direct = _as_str(payload.get("text") or payload.get("combined_text"))
    field_values = {field: _as_str(payload.get(field)) for field in TEXT_FIELDS}
    has_fields = any(field_values.values())
    if not direct and not has_fields:
        raise ValueError(
            "Provide 'text' / 'combined_text', or at least one of: "
            + ", ".join(TEXT_FIELDS)
        )
    if has_fields:
        combined_parts = [field_values[f] for f in TEXT_FIELDS if field_values[f]]
        combined_text = "\n".join(combined_parts)
        tagged_parts = []
        tag_map = {
            "title": "TITLE",
            "company_profile": "COMPANY PROFILE",
            "description": "DESCRIPTION",
            "requirements": "REQUIREMENTS",
            "benefits": "BENEFITS",
        }
        for field in TEXT_FIELDS:
            if field_values[field]:
                tagged_parts.append(f"[{tag_map[field]}] {field_values[field]}")
        model_text = "\n".join(tagged_parts)
    else:
        combined_text = direct
        lines = [ln.strip() for ln in direct.splitlines() if ln.strip()]
        if not lines:
            model_text = ""
        elif len(lines) == 1:
            model_text = f"[TITLE] {lines[0]}"
        else:
            model_text = f"[TITLE] {lines[0]}\n[DESCRIPTION] " + "\n".join(lines[1:])
    if not combined_text.strip():
        raise ValueError("Resolved text is empty")
    _enforce_text_length(combined_text, field="combined_text")
    resolved_model = model_text or combined_text
    _enforce_text_length(resolved_model, field="model_text")
    return {
        "record_id": record_id,
        "combined_text": combined_text,
        "model_text": resolved_model,
    }
