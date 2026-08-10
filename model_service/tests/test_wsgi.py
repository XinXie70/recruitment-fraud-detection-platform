from __future__ import annotations

import runpy
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def test_wsgi_bootstraps_models_before_exporting_app(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_app_module = ModuleType("app")
    fake_app_module.app = MagicMock()
    fake_app_module.lr_service = MagicMock()
    fake_app_module.bert_service = MagicMock()

    fake_settings_module = ModuleType("settings")
    fake_settings_module.load_runtime_config = lambda: {"ready": True}

    monkeypatch.setitem(sys.modules, "app", fake_app_module)
    monkeypatch.setitem(sys.modules, "settings", fake_settings_module)
    monkeypatch.setenv("ALLOW_CPU", "false")

    namespace = runpy.run_path("model_service/wsgi.py")

    fake_app_module.lr_service.load.assert_called_once_with()
    fake_app_module.bert_service.load.assert_called_once_with(allow_cpu=False)
    assert namespace["app"] is fake_app_module.app
    assert namespace["__all__"] == ["app"]
    assert "Models ready." in capsys.readouterr().out
