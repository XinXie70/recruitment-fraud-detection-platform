#Flask API for LR, BERT, FP-gate ensemble, risk score and risk level
from __future__ import annotations
import logging
import os
from flasgger import Swagger
from flask import Flask
from flask_cors import CORS
from settings import (
    MAX_CONTENT_LENGTH,
    MODEL_API_KEY,
    MODEL_CORS_ORIGINS,
    load_runtime_config,
)
from services.bert_service import bert_service
from services.lr_service import lr_service
from api_http import register_http_handlers
from prediction_routes import routes
from extensions import limiter
logger = logging.getLogger("model_service")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["MODEL_API_KEY"] = MODEL_API_KEY
CORS(app, origins=MODEL_CORS_ORIGINS)
limiter.init_app(app)

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
register_http_handlers(app)
app.register_blueprint(routes)
if not MODEL_API_KEY:
    logger.warning(
        "MODEL_API_KEY is unset; /predict/* authentication is disabled. "
        "Set MODEL_API_KEY before exposing this service."
    )
def create_app() -> Flask:
    return app
if __name__ == "__main__":
    print("Loading runtime config...")
    print(load_runtime_config())
    print("Loading LR...")
    lr_service.load()
    print("Loading BERT (first request may still warm CUDA kernels)...")
    bert_service.load()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    print(f"Swagger UI: http://127.0.0.1:{port}/apidocs/")
    app.run(host=host, port=port, debug=False, threaded=True)
