from __future__ import annotations

import httpx
import pytest

from services.ensemble_predictor import EnsembleUnavailableError
from services.remote_ensemble_predictor import RemoteFinalEnsemblePredictor


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
            "thresholds": {
                "bert_low_threshold": 0.15,
                "bert_high_threshold": 0.32,
                "lr_gate": 0.2,
            },
        },
    }


def test_remote_predictor_maps_final_ensemble_contract() -> None:
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

    predictor = RemoteFinalEnsemblePredictor("http://model/")
    predictor.client = httpx.Client(transport=httpx.MockTransport(handler))

    computation = predictor.predict("job listing")

    assert computation.ensemble.risk_score == pytest.approx(0.88)
    assert computation.ensemble.risk_score_100 == pytest.approx(88.0)
    assert computation.ensemble.risk_level == "high"
    assert computation.ensemble.prediction == "fake"
    assert computation.ensemble.active_model_count == 2
    assert [member.key for member in computation.members] == ["lr", "bert"]
    assert computation.members[1].effective_weight == 1
    assert computation.score_batch(["one", "two"]) == [0.2, 0.7]
    assert [request.url.path for request in requests] == [
        "/predict/all",
        "/predict/batch",
    ]


def test_remote_predictor_rejects_contract_mismatch() -> None:
    predictor = RemoteFinalEnsemblePredictor("http://model")
    predictor.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"ok": True})
        )
    )

    with pytest.raises(EnsembleUnavailableError, match="API contract"):
        predictor.predict("job listing")
