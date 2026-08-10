FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --no-create-home app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY --chown=app:app backend /app/backend
COPY --chown=app:app model /app/model
COPY --chown=app:app model_algorithm/sprint3/LR/weight /app/model_algorithm/sprint3/LR/weight
COPY --chown=app:app model_algorithm/sprint3/LR/results/config.json /app/model_algorithm/sprint3/LR/results/config.json
COPY --chown=app:app model_algorithm/sprint3/BERT/weight/best /app/model_algorithm/sprint3/BERT/weight/best
COPY --chown=app:app model_algorithm/sprint3/ensemble_BERT_FP/results/config.json /app/model_algorithm/sprint3/ensemble_BERT_FP/results/config.json
COPY --chown=app:app model_algorithm/sprint3/risk_level/risk_boundary_config.json /app/model_algorithm/sprint3/risk_level/risk_boundary_config.json
RUN pip install --no-cache-dir --no-deps --editable /app/model

USER app

EXPOSE 8000

# Render provides $PORT env var; fall back to 8000 for local dev
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
