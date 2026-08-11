from __future__ import annotations

from flask import Blueprint, jsonify

from api_http import error_response as _error, parse_payload as _parse_payload, server_error as _server_error, with_meta as _with_meta
from settings import (
    MAX_BATCH_ITEMS,
    MAX_BATCH_TOTAL_CHARS,
    RATE_LIMIT_PREDICT,
    RATE_LIMIT_PREDICT_BATCH,
    load_runtime_config,
)
from services.bert_service import bert_service
from services.ensemble_service import ensemble_service, risk_service
from services.lr_service import lr_service
from services.text_utils import build_texts

from extensions import limiter

routes = Blueprint("prediction_api", __name__)

@routes.get("/health")
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


@routes.get("/config")
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
@routes.post("/predict/lr")
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


@routes.post("/predict/bert")
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


@routes.post("/predict/ensemble")
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


@routes.post("/predict/risk")
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


@routes.post("/predict/all")
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


@routes.post("/predict/batch")
@limiter.limit(RATE_LIMIT_PREDICT_BATCH)
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
