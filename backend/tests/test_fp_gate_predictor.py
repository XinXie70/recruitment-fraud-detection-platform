from __future__ import annotations

import httpx
import pytest

from backend.services.fp_gate_predictor import EnsembleUnavailableError, FPGatePredictor



def _all_response() -> dict:
    return {
        "ok": True,
        "lr": {"lr_score": 0.72},
        "bert": {"bert_score": 0.88},
        "ensemble": {
            "model": "ensemble_fp_gate",
            "predicted_label_id": 1,
        },
        "risk": {
            "risk_score": 0.88,
            "risk_score_100": 88.0,
            "risk_level": "High",
            "risk_score_source": "bert",
            "gate_triggered": False,
            "decision_reason": "BERT High rule passed the LR gate",
            "thresholds": {
                "bert_low_threshold": 0.15,
                "bert_high_threshold": 0.32,
                "lr_gate": 0.2,
            },
        },
    }


def test_fp_gate_predictor_maps_final_ensemble_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/predict/all":
            return httpx.Response(200, json=_all_response())
        return httpx.Response(
            200,
            json={
                "ok": True,
                "count": 2,
                "results": [{"risk_score": 0.2}, {"risk_score": 0.7}],
            },
        )

    predictor = FPGatePredictor("http://model/")
    predictor.client = httpx.Client(transport=httpx.MockTransport(handler))

    computation = predictor.predict("job listing")

    assert computation.ensemble.risk_score == pytest.approx(0.88)
    assert computation.ensemble.risk_score_100 == pytest.approx(88.0)
    assert computation.ensemble.risk_level == "high"
    assert computation.ensemble.prediction == "fake"
    assert computation.ensemble.active_model_count == 2
    assert [member.key for member in computation.members] == ["lr", "bert"]
    assert computation.members[0].role == "false_positive_gate"
    assert computation.members[1].role == "primary_score"
    assert computation.members[1].decision_active is True
    assert computation.ensemble.method == "bert_lr_fp_gate"
    assert computation.ensemble.risk_score_source == "bert"
    assert computation.ensemble.gate_triggered is False
    assert computation.ensemble.lr_gate_threshold == pytest.approx(0.2)
    assert computation.score_batch(["one", "two"]) == [0.2, 0.7]
    assert [request.url.path for request in requests] == [
        "/predict/all",
        "/predict/batch",
    ]


def test_fp_gate_predictor_rejects_contract_mismatch() -> None:
    predictor = FPGatePredictor("http://model")
    predictor.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"ok": True})
        )
    )

    with pytest.raises(EnsembleUnavailableError, match="API contract"):
        predictor.predict("job listing")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(503), "request failed"),
        (httpx.Response(200, json={"ok": False}), "returned an error"),
        (httpx.Response(200, json=["unexpected"]), "returned an error"),
    ],
)
def test_fp_gate_predictor_rejects_request_failures(
    response: httpx.Response, message: str
) -> None:
    predictor = FPGatePredictor("http://model")
    predictor.client = httpx.Client(
        transport=httpx.MockTransport(lambda request: response)
    )

    with pytest.raises(EnsembleUnavailableError, match=message):
        predictor.predict("job listing")


@pytest.mark.parametrize("value", ["not-a-number", -0.1, 1.1])
def test_fp_gate_predictor_rejects_invalid_probabilities(value: object) -> None:
    payload = _all_response()
    payload["risk"]["risk_score"] = value
    predictor = FPGatePredictor("http://model")
    predictor.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)
        )
    )

    with pytest.raises(EnsembleUnavailableError, match="invalid risk_score"):
        predictor.predict("job listing")


def test_fp_gate_predictor_warm_up_reports_success_and_failure() -> None:
    predictor = FPGatePredictor("http://model")
    predictor.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_all_response())
        )
    )
    assert predictor.warm_up("job listing") == {"final_ensemble": None}

    predictor.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503)
        )
    )
    outcome = predictor.warm_up("job listing")
    assert outcome["final_ensemble"] == "FP-gate model service request failed."


@pytest.mark.parametrize(
    ("response", "available"),
    [
        (httpx.Response(200, json={"ok": True}), True),
        (httpx.Response(200, json={"ok": False}), False),
        (httpx.Response(503), False),
    ],
)
def test_fp_gate_live_health_probe(response: httpx.Response, available: bool) -> None:
    predictor = FPGatePredictor("http://model")
    predictor.client = httpx.Client(
        transport=httpx.MockTransport(lambda request: response)
    )

    assert predictor.is_available() is available


def test_fp_gate_predictor_rejects_batch_contract_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/predict/all":
            return httpx.Response(200, json=_all_response())
        return httpx.Response(200, json={"ok": True, "results": []})

    predictor = FPGatePredictor("http://model")
    predictor.client = httpx.Client(transport=httpx.MockTransport(handler))
    computation = predictor.predict("job listing")

    with pytest.raises(EnsembleUnavailableError, match="batch response"):
        computation.score_batch(["one"])

    predictor.close()
