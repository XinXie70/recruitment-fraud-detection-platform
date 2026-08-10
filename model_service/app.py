"""Flask API for LR, BERT, FP-gate ensemble, risk score and risk level.

Swagger UI: http://127.0.0.1:5000/apidocs/
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any

from flasgger import Swagger
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from settings import (
    MAX_BATCH_ITEMS,
    MAX_BATCH_TOTAL_CHARS,
    MAX_CONTENT_LENGTH,
    MODEL_API_KEY,
    RATE_LIMIT_PREDICT,
    load_runtime_config,
)
from services.bert_service import bert_service
from services.ensemble_service import ensemble_service, risk_service
from services.lr_service import lr_service
from services.text_utils import ResolvedTexts, build_texts

logger = logging.getLogger("model_service")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

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
            "and formal risk score / risk level for frontend or backend clients. "
            "Predict endpoints require header X-API-Key when MODEL_API_KEY is set."
        ),
        "version": "1.0.0",
    },
    "basePath": "/",
    "schemes": ["http"],
    "securityDefinitions": {
        "ApiKeyAuth": {
            "type": "apiKey",
            "name": "X-API-Key",
            "in": "header",
        }
    },
    "tags": [
        {"name": "system", "description": "Health and configuration"},
        {"name": "predict", "description": "Model and risk prediction endpoints"},
    ],
}

Swagger(app, config=swagger_config, template=swagger_template)

if not MODEL_API_KEY:
    logger.warning(
        "MODEL_API_KEY is unset; /predict/* authentication is disabled. "
        "Set MODEL_API_KEY before exposing this service."
    )


def _error(message: str, status: int = 400, details: str | None = None):
    payload: dict[str, Any] = {"ok": False, "error": message}
    if details:
        payload["details"] = details
    return jsonify(payload), status


def _server_error(exc: Exception):
    logger.exception("Unhandled prediction error: %s", exc)
    return _error("Internal server error", status=500)


def _parse_payload() -> dict[str, Any]:
    if not request.is_json:
        raise ValueError("Content-Type must be application/json")
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValueError("Invalid or empty JSON body")
    return payload


def _with_meta(result: dict[str, Any], texts: ResolvedTexts) -> dict[str, Any]:
    out = {"ok": True, **result}
    if texts.get("record_id"):
        out["record_id"] = texts["record_id"]
    return out


def _extract_api_key() -> str | None:
    header_key = request.headers.get("X-API-Key")
    if header_key:
        return header_key.strip()
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


@app.before_request
def require_api_key():
    if not request.path.startswith("/predict"):
        return None
    if not MODEL_API_KEY:
        return None
    provided = _extract_api_key()
    if not provided or not secrets.compare_digest(provided, MODEL_API_KEY):
        return _error("Unauthorized", status=401)
    return None


@app.errorhandler(413)
def request_entity_too_large(_exc):
    return _error(
        f"Request body exceeds limit of {MAX_CONTENT_LENGTH} bytes",
        status=413,
    )


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
        return _server_error(exc)


@app.post("/predict/lr")
@limiter.limit(RATE_LIMIT_PREDICT)
def predict_lr():
    """Predict with Logistic Regression only.
    ---
    tags:
      - predict
    security:
      - ApiKeyAuth: []
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
      401:
        description: Unauthorized
      429:
        description: Rate limited
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
        return _server_error(exc)


@app.post("/predict/bert")
@limiter.limit(RATE_LIMIT_PREDICT)
def predict_bert():
    """Predict with BERT (max_length=512) only.
    ---
    tags:
      - predict
    security:
      - ApiKeyAuth: []
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
      401:
        description: Unauthorized
      429:
        description: Rate limited
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
        return _server_error(exc)


@app.post("/predict/ensemble")
@limiter.limit(RATE_LIMIT_PREDICT)
def predict_ensemble():
    """Predict with BERT + LR FP-gate ensemble.
    ---
    tags:
      - predict
    security:
      - ApiKeyAuth: []
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
      401:
        description: Unauthorized
      429:
        description: Rate limited
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
        return _server_error(exc)


@app.post("/predict/risk")
@limiter.limit(RATE_LIMIT_PREDICT)
def predict_risk():
    """Return formal risk_score and risk_level.
    ---
    tags:
      - predict
    security:
      - ApiKeyAuth: []
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
      401:
        description: Unauthorized
      429:
        description: Rate limited
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
        return _server_error(exc)


@app.post("/predict/all")
@limiter.limit(RATE_LIMIT_PREDICT)
def predict_all():
    """Return LR, BERT, ensemble, and risk outputs in one response.
    ---
    tags:
      - predict
    security:
      - ApiKeyAuth: []
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
      401:
        description: Unauthorized
      429:
        description: Rate limited
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
        return _server_error(exc)


@app.post("/predict/batch")
@limiter.limit(RATE_LIMIT_PREDICT)
def predict_batch():
    """Batch risk prediction for multiple advertisements.
    ---
    tags:
      - predict
    security:
      - ApiKeyAuth: []
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
      401:
        description: Unauthorized
      429:
        description: Rate limited
      500:
        description: Model failure
    """
    try:
        payload = _parse_payload()
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("'items' must be a non-empty array")
        if len(items) > MAX_BATCH_ITEMS:
            raise ValueError(f"Batch size limited to {MAX_BATCH_ITEMS} items")

        results = []
        total_chars = 0
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"items[{idx}] must be an object")
            texts = build_texts(item)
            total_chars += len(texts["combined_text"]) + len(texts["model_text"])
            if total_chars > MAX_BATCH_TOTAL_CHARS:
                raise ValueError(
                    f"Batch total character count exceeds {MAX_BATCH_TOTAL_CHARS}"
                )
            risk = risk_service.predict(texts["combined_text"], texts["model_text"])
            results.append(_with_meta(risk, texts))
        return jsonify({"ok": True, "count": len(results), "results": results})
    except ValueError as exc:
        return _error(str(exc), status=400)
    except Exception as exc:  # noqa: BLE001
        return _server_error(exc)


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    # Eager-load models so the first HTTP call is not cold.
    print("Loading runtime config...")
    print(load_runtime_config())
    print("Loading LR...")
    lr_service.load()
    print("Loading BERT (first request may still warm CUDA kernels)...")
    bert_service.load()
    # Default to localhost for local runs; Docker sets HOST=0.0.0.0 explicitly.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    print(f"Swagger UI: http://127.0.0.1:{port}/apidocs/")
    app.run(host=host, port=port, debug=False, threaded=True)
