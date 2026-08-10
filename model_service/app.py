"""Flask API for LR, BERT, FP-gate ensemble, risk score and risk level.

Swagger UI: http://127.0.0.1:5000/apidocs/
"""

from __future__ import annotations

import traceback
from typing import Any

from flasgger import Swagger
from flask import Flask, jsonify, request

from settings import load_runtime_config
from services.bert_service import bert_service
from services.ensemble_service import ensemble_service, risk_service
from services.lr_service import lr_service
from services.text_utils import TextInputs, build_texts

app = Flask(__name__)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Fraud Job Ad Detection API",
        "description": (
            "Serve LR (bigram/no-CV), BERT (maxlen=512), FP-gate ensemble, "
            "and formal risk score / risk level for frontend or backend clients."
        ),
        "version": "1.0.0",
    },
    "basePath": "/",
    "schemes": ["http"],
    "tags": [
        {"name": "system", "description": "Health and configuration"},
        {"name": "predict", "description": "Model and risk prediction endpoints"},
    ],
}

Swagger(app, config=swagger_config, template=swagger_template)

PREDICT_BODY = {
    "text": {
        "type": "string",
        "description": "Free-form job advertisement text (combined_text).",
        "example": (
            "Urgent work from home job. Send your bank details and passport "
            "copy to apply today. High salary guaranteed."
        ),
    },
    "combined_text": {
        "type": "string",
        "description": "Alias of text.",
    },
    "record_id": {
        "type": "string",
        "description": "Optional caller-side record id.",
        "example": "demo_001",
    },
    "title": {"type": "string"},
    "company_profile": {"type": "string"},
    "description": {"type": "string"},
    "requirements": {"type": "string"},
    "benefits": {"type": "string"},
}


def _error(message: str, status: int = 400, details: str | None = None):
    payload: dict[str, Any] = {"ok": False, "error": message}
    if details:
        payload["details"] = details
    return jsonify(payload), status


def _parse_payload() -> dict[str, Any]:
    if not request.is_json:
        raise ValueError("Content-Type must be application/json")
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValueError("Invalid or empty JSON body")
    return payload


def _with_meta(result: dict[str, Any], texts: TextInputs) -> dict[str, Any]:
    out = {"ok": True, **result}
    if texts.get("record_id"):
        out["record_id"] = texts["record_id"]
    return out


@app.get("/health")
def health():
    """Service health check.
    ---
    tags:
      - system
    responses:
      200:
        description: Service is up
    """
    return jsonify({"ok": True, "status": "healthy"})


@app.get("/config")
def get_config():
    """Return frozen thresholds and model paths used by the API.
    ---
    tags:
      - system
    responses:
      200:
        description: Runtime configuration
      500:
        description: Config files missing
    """
    try:
        return jsonify({"ok": True, "config": load_runtime_config()})
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), status=500)


@app.post("/predict/lr")
def predict_lr():
    """Predict with Logistic Regression only.
    ---
    tags:
      - predict
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
            combined_text:
              type: string
            record_id:
              type: string
            title:
              type: string
            company_profile:
              type: string
            description:
              type: string
            requirements:
              type: string
            benefits:
              type: string
    responses:
      200:
        description: LR prediction
      400:
        description: Bad request
      500:
        description: Model failure
    """
    try:
        texts = build_texts(_parse_payload())
        result = lr_service.predict(texts["combined_text"])
        return jsonify(_with_meta(result, texts))
    except ValueError as exc:
        return _error(str(exc), status=400)
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), status=500, details=traceback.format_exc())


@app.post("/predict/bert")
def predict_bert():
    """Predict with BERT (max_length=512) only.
    ---
    tags:
      - predict
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
            combined_text:
              type: string
            record_id:
              type: string
            title:
              type: string
            company_profile:
              type: string
            description:
              type: string
            requirements:
              type: string
            benefits:
              type: string
    responses:
      200:
        description: BERT prediction
      400:
        description: Bad request
      500:
        description: Model failure
    """
    try:
        texts = build_texts(_parse_payload())
        result = bert_service.predict(texts["model_text"])
        return jsonify(_with_meta(result, texts))
    except ValueError as exc:
        return _error(str(exc), status=400)
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), status=500, details=traceback.format_exc())


@app.post("/predict/ensemble")
def predict_ensemble():
    """Predict with BERT + LR FP-gate ensemble.
    ---
    tags:
      - predict
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
            combined_text:
              type: string
            record_id:
              type: string
            title:
              type: string
            company_profile:
              type: string
            description:
              type: string
            requirements:
              type: string
            benefits:
              type: string
    responses:
      200:
        description: Ensemble prediction
      400:
        description: Bad request
      500:
        description: Model failure
    """
    try:
        texts = build_texts(_parse_payload())
        result = ensemble_service.predict(
            texts["combined_text"], texts["model_text"]
        )
        return jsonify(_with_meta(result, texts))
    except ValueError as exc:
        return _error(str(exc), status=400)
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), status=500, details=traceback.format_exc())


@app.post("/predict/risk")
def predict_risk():
    """Return formal risk_score and risk_level.
    ---
    tags:
      - predict
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
            combined_text:
              type: string
            record_id:
              type: string
            title:
              type: string
            company_profile:
              type: string
            description:
              type: string
            requirements:
              type: string
            benefits:
              type: string
    responses:
      200:
        description: Risk score and level
      400:
        description: Bad request
      500:
        description: Model failure
    """
    try:
        texts = build_texts(_parse_payload())
        result = risk_service.predict(texts["combined_text"], texts["model_text"])
        return jsonify(_with_meta(result, texts))
    except ValueError as exc:
        return _error(str(exc), status=400)
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), status=500, details=traceback.format_exc())


@app.post("/predict/all")
def predict_all():
    """Return LR, BERT, ensemble, and risk outputs in one response.
    ---
    tags:
      - predict
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              example: Urgent work from home job. Send bank details to apply.
            combined_text:
              type: string
            record_id:
              type: string
              example: demo_001
            title:
              type: string
            company_profile:
              type: string
            description:
              type: string
            requirements:
              type: string
            benefits:
              type: string
    responses:
      200:
        description: Combined prediction payload
      400:
        description: Bad request
      500:
        description: Model failure
    """
    try:
        texts = build_texts(_parse_payload())
        lr = lr_service.predict(texts["combined_text"])
        bert = bert_service.predict(texts["model_text"])
        ensemble = ensemble_service.predict(
            texts["combined_text"],
            texts["model_text"],
            lr_score=lr["lr_score"],
            bert_score=bert["bert_score"],
        )
        risk = risk_service.predict(
            texts["combined_text"],
            texts["model_text"],
            lr_score=lr["lr_score"],
            bert_score=bert["bert_score"],
        )
        payload = {
            "ok": True,
            "record_id": texts.get("record_id"),
            "lr": lr,
            "bert": bert,
            "ensemble": ensemble,
            "risk": risk,
        }
        return jsonify(payload)
    except ValueError as exc:
        return _error(str(exc), status=400)
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), status=500, details=traceback.format_exc())


@app.post("/predict/batch")
def predict_batch():
    """Batch risk prediction for multiple advertisements.
    ---
    tags:
      - predict
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - items
          properties:
            items:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                  record_id:
                    type: string
                  title:
                    type: string
                  description:
                    type: string
    responses:
      200:
        description: Batch results
      400:
        description: Bad request
      500:
        description: Model failure
    """
    try:
        payload = _parse_payload()
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("'items' must be a non-empty array")
        if len(items) > 100:
            raise ValueError("Batch size limited to 100 items")

        results = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"items[{idx}] must be an object")
            texts = build_texts(item)
            risk = risk_service.predict(texts["combined_text"], texts["model_text"])
            results.append(_with_meta(risk, texts))
        return jsonify({"ok": True, "count": len(results), "results": results})
    except ValueError as exc:
        return _error(str(exc), status=400)
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), status=500, details=traceback.format_exc())


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    import os

    # Eager-load models so the first HTTP call is not cold.
    print("Loading runtime config...")
    print(load_runtime_config())
    print("Loading LR...")
    lr_service.load()
    print("Loading BERT (first request may still warm CUDA kernels)...")
    bert_service.load()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    print(f"Swagger UI: http://127.0.0.1:{port}/apidocs/")
    app.run(host=host, port=port, debug=False, threaded=True)
