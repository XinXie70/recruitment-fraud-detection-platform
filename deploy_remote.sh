
set -e

PORT=${1:-8001}

echo "=== Installing dependencies ==="
pip install -r backend/requirements-remote.txt

echo "=== Starting backend (port $PORT) ==="
ENVIRONMENT=production \
DATABASE_URL='postgresql+psycopg2://postgres.xhxqgsjvrirhpefeayix:ww.SPZ34n%2FtT28%40@aws-1-ap-south-1.pooler.supabase.com:6543/postgres' \
SECRET_KEY='ezlJo5Zn3TdQ34ReY6YUyC1orPCb5QqCydjdYJisqmg719dRwZqXS8xha9UNMLj3vHOSVIaR3kCB62rFIkgUxQ' \
ACCESS_TOKEN_EXPIRE_MINUTES=10080 \
CORS_ORIGINS='http://127.0.0.1:5190,http://localhost:5190' \
MAX_INPUT_CHARS=50000 \
MODEL_TIMEOUT_SECONDS=30 \
MODEL_MAX_WORKERS=3 \
MODEL_SERVER_URL='http://127.0.0.1:8000' \
MODEL_SERVER_TIMEOUT=120 \
gunicorn backend.main:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:$PORT \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
