from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import get_current_user
from routers.analysis import get_analysis_service, router

from test_analysis_contract import build_service


def test_versioned_analysis_endpoint_matches_the_response_contract():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_analysis_service] = build_service
    client = TestClient(app)

    response = client.post(
        "/api/v1/analyze",
        json={"text": "Urgent job. Pay a fee before the interview."},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "api_version",
        "status",
        "job_relevance_score",
        "ensemble",
        "member_outputs",
        "xai",
        "gentle_ai",
        "url_analysis",
    }
    assert set(payload["ensemble"]) == {
        "status",
        "risk_score",
        "classification_label",
        "risk_level",
        "prediction",
        "recommended_action",
        "low_threshold",
        "high_threshold",
        "active_model_count",
        "failed_model_count",
        "version",
        "fitted",
        "weight_source",
    }
    assert set(payload["member_outputs"][0]) == {
        "key",
        "display_name",
        "status",
        "raw_score",
        "calibrated_score",
        "configured_weight",
        "effective_weight",
        "weighted_contribution",
        "error",
        "error_code",
    }
    assert set(payload["xai"]) == {
        "status",
        "method",
        "target",
        "version",
        "base_value",
        "output_value",
        "items",
        "message",
    }
    assert set(payload["gentle_ai"]) == {
        "status",
        "provider",
        "summary",
        "evidence_explanations",
        "next_steps",
        "learning_item_ids",
        "disclaimer",
        "message",
        "version",
    }


def test_legacy_predict_path_uses_the_same_backend_contract():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_analysis_service] = build_service
    client = TestClient(app)

    response = client.post(
        "/api/predict",
        json={"text": "Urgent job. Pay a fee before the interview."},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 200
    assert response.json()["api_version"] == "1.0"
    assert response.json()["ensemble"]["risk_level"] == "high"


def test_main_module_imports_without_loading_model_artifacts():
    import main

    assert main.app.version == "2.0.0"
