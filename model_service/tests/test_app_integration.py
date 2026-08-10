from __future__ import annotations

import pytest


def test_health_is_open(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "status": "healthy"}


def test_config_returns_runtime_paths(client) -> None:
    response = client.get("/config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert "lr" in payload["config"]
    assert "ensemble" in payload["config"]


def test_predict_all_happy_path(client, sample_payload, mock_model_services) -> None:
    response = client.post("/predict/all", json=sample_payload)

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["record_id"] == "demo_001"
    assert "lr" in payload and "bert" in payload and "risk" in payload
    mock_model_services["lr"].predict.assert_called_once()
    mock_model_services["bert"].predict.assert_called_once()


def test_predict_all_requires_json_body(client) -> None:
    response = client.post("/predict/all", data="plain-text")

    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_predict_all_rejects_blank_payload(client) -> None:
    response = client.post("/predict/all", json={"title": "   "})

    assert response.status_code == 400
    assert "Provide 'text'" in response.get_json()["error"]


def test_predict_batch_happy_path(client, sample_payload, mock_model_services) -> None:
    response = client.post(
        "/predict/batch",
        json={"items": [sample_payload, {"text": "Legitimate office role."}]},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["count"] == 2
    assert len(payload["results"]) == 2
    assert mock_model_services["risk"].predict.call_count == 2


@pytest.mark.parametrize(
    "body,expected_fragment",
    [
        ({"items": []}, "non-empty array"),
        ({"items": "bad"}, "non-empty array"),
        ({"items": ["bad"]}, "must be an object"),
    ],
)
def test_predict_batch_validation_errors(client, body, expected_fragment: str) -> None:
    response = client.post("/predict/batch", json=body)

    assert response.status_code == 400
    assert expected_fragment in response.get_json()["error"]


def test_predict_batch_item_limit(client, monkeypatch: pytest.MonkeyPatch) -> None:
    import prediction_routes

    monkeypatch.setattr(prediction_routes, "MAX_BATCH_ITEMS", 1)

    response = client.post(
        "/predict/batch",
        json={"items": [{"text": "one"}, {"text": "two"}]},
    )

    assert response.status_code == 400
    assert "Batch size limited" in response.get_json()["error"]


def test_predict_requires_api_key_when_configured(authed_client, sample_payload) -> None:
    response = authed_client.post("/predict/all", json=sample_payload)

    assert response.status_code == 401


def test_predict_accepts_x_api_key_header(authed_client, sample_payload) -> None:
    response = authed_client.post(
        "/predict/all",
        json=sample_payload,
        headers={"X-API-Key": "test-secret-key"},
    )

    assert response.status_code == 200
    assert response.get_json()["ok"] is True


def test_predict_accepts_bearer_token(authed_client, sample_payload) -> None:
    response = authed_client.post(
        "/predict/all",
        json=sample_payload,
        headers={"Authorization": "Bearer test-secret-key"},
    )

    assert response.status_code == 200


def test_predict_rejects_wrong_api_key(authed_client, sample_payload) -> None:
    response = authed_client.post(
        "/predict/all",
        json=sample_payload,
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401
    assert response.get_json()["error"] == "Unauthorized"


def test_model_failure_returns_sanitized_500(
    client,
    sample_payload,
    mock_model_services,
) -> None:
    mock_model_services["lr"].predict.side_effect = RuntimeError("boom")

    response = client.post("/predict/all", json=sample_payload)

    assert response.status_code == 500
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"] == "Internal server error"
    assert "traceback" not in response.get_data(as_text=True).lower()
    assert "boom" not in response.get_data(as_text=True)
