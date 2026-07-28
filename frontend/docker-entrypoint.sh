#!/bin/sh
set -e

# Substitute ${BACKEND_URL} in nginx config template
envsubst '${BACKEND_URL}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf

echo "Starting nginx with BACKEND_URL=${BACKEND_URL}"

exec "$@"
