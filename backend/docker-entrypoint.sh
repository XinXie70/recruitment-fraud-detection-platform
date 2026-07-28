#!/bin/sh
set -eu

if [ "${RUN_DATABASE_MIGRATIONS:-false}" = "true" ]; then
    python -m alembic -c /app/backend/alembic.ini upgrade head
fi

exec "$@"
