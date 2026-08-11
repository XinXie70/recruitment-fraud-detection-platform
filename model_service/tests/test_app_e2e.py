from __future__ import annotations
import importlib
import os
from pathlib import Path
import pytest
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
ARTIFACTS_PRESENT = (
    (MODELS_ROOT / "lr" / "lr_none_bigram_no_cv_paper_aligned_seed42.joblib").exists()
    and (MODELS_ROOT / "bert" / "best" / "model.safetensors").exists()
)
pytestmark = pytest.mark.e2e
@pytest.mark.skipif(
    os.getenv("RUN_MODEL_SERVICE_E2E") != "1",
    reason="Set RUN_MODEL_SERVICE_E2E=1 to run slow end-to-end model_service tests.",
)
@pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="Frozen model artifacts are not present under model_service/models/.",
)
def test_e2e_predict_all_with_real_weights(sample_payload: dict[str, str]) -> None:
    import settings
    importlib.reload(settings)
    from services.bert_service import BERTService
    from services.lr_service import LRService
    lr = LRService()
    bert = BERTService()
    lr.load()
    bert.load(allow_cpu=True)
    import app as app_module
    import prediction_routes
    prediction_routes.lr_service = lr
    prediction_routes.bert_service = bert
    client = app_module.app.test_client()
    response = client.post("/predict/all", json=sample_payload)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert 0.0 <= payload["lr"]["lr_score"] <= 1.0
    assert 0.0 <= payload["bert"]["bert_score"] <= 1.0
    assert payload["risk"]["risk_level"] in {"Low", "Suspicious", "High"}
