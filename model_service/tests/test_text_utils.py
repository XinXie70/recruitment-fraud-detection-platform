from __future__ import annotations
import pytest
from services.text_utils import build_texts
def test_build_texts_from_free_text(sample_text: str) -> None:
    texts = build_texts({"text": sample_text, "record_id": "42"})
    assert texts["record_id"] == "42"
    assert sample_text in texts["combined_text"]
    assert texts["model_text"].startswith("[TITLE]")
def test_build_texts_from_structured_fields() -> None:
    texts = build_texts(
        {
            "title": "Remote assistant",
            "description": "Flexible hours.",
            "requirements": "Must have laptop.",
        }
    )
    assert "Remote assistant" in texts["combined_text"]
    assert "[TITLE] Remote assistant" in texts["model_text"]
    assert "[DESCRIPTION] Flexible hours." in texts["model_text"]

def test_build_texts_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="Provide 'text'"):
        build_texts({})

def test_build_texts_rejects_non_object_payload() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        build_texts("not-an-object")  
def test_build_texts_rejects_blank_payload_fields() -> None:
    with pytest.raises(ValueError, match="Provide 'text'"):
        build_texts({"title": "   ", "description": "nan"})
def test_build_texts_rejects_overlong_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_TEXT_CHARS", "20")
    import importlib
    import settings
    importlib.reload(settings)
    import services.text_utils as text_utils
    importlib.reload(text_utils)
    with pytest.raises(ValueError, match="maximum length"):
        text_utils.build_texts({"text": "x" * 21})
def test_build_texts_normalizes_null_like_values() -> None:
    texts = build_texts({"title": "Real title", "description": "null"})
    assert texts["combined_text"] == "Real title"
    assert "[TITLE] Real title" in texts["model_text"]
