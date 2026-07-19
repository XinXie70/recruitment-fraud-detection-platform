#!/usr/bin/env bash
# ============================================================
# Production one-click deployment script — Backend + Frontend + Database
# (Model inference is split to a separate server; this script does not deploy model services)
#
# Usage: chmod +x deploy.sh && ./deploy.sh
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Fake Job Detection — Backend+Frontend+DB Deployment${NC}"
echo -e "${GREEN}============================================${NC}"

# 1. Check .env file
if [ ! -f .env ]; then
    if [ -f .env.prod ]; then
        echo -e "${YELLOW}[!] .env not found, copying from .env.prod...${NC}"
        cp .env.prod .env
        echo -e "${RED}[⚠] Please edit .env first — update DB password and SECRET_KEY!${NC}"
        echo -e "${RED}    If using a remote model server, also set MODEL_SERVER_URL${NC}"
        echo -e "${RED}    After editing, re-run: ./deploy.sh${NC}"
        exit 1
    else
        echo -e "${RED}[✗] .env.prod not found either. Please create .env first${NC}"
        exit 1
    fi
fi

# 2. Check Docker
if ! command -v docker &>/dev/null; then
    echo -e "${RED}[✗] Docker not found. Please install Docker first${NC}"
    exit 1
fi

if ! docker compose version &>/dev/null && ! docker-compose version &>/dev/null; then
    echo -e "${RED}[✗] Docker Compose not found. Please install it first${NC}"
    exit 1
fi

# 3. Determine compose command
if docker compose version &>/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# 4. Detect model server config and select the right Dockerfile
MODEL_SERVER_URL=$(grep -E '^MODEL_SERVER_URL=' .env 2>/dev/null | cut -d= -f2- | tr -d '[:space:]' || echo "")
if [ -n "$MODEL_SERVER_URL" ]; then
    echo -e "${GREEN}[✓] MODEL_SERVER_URL=${MODEL_SERVER_URL} — using remote model server${NC}"
    export BACKEND_DOCKERFILE=backend/Dockerfile.remote
else
    echo -e "${YELLOW}[!] MODEL_SERVER_URL not set — loading models locally (needs 8GB+ RAM)${NC}"
    export BACKEND_DOCKERFILE=backend/Dockerfile
fi

# 5. Pull base images
echo -e "${GREEN}[1/5] Pulling base images...${NC}"
docker pull postgres:16-alpine
docker pull nginx:1.27-alpine

# 6. Build images
echo -e "${GREEN}[2/5] Building backend image (${BACKEND_DOCKERFILE})...${NC}"
$COMPOSE_CMD -f docker-compose.prod.yml build backend

echo -e "${GREEN}[3/5] Building frontend image...${NC}"
$COMPOSE_CMD -f docker-compose.prod.yml build frontend

# 7. Start services

echo -e "${GREEN}[4/5] Starting all services...${NC}"
$COMPOSE_CMD -f docker-compose.prod.yml up -d

# 8. Wait for health check

echo -e "${GREEN}[5/5] Waiting for services to be ready...${NC}"
sleep 5

# Show status
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Deployment complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "Frontend:     http://localhost"
echo -e "Backend API:  http://localhost:8000"
echo -e "API Docs:     http://localhost:8000/docs"
echo -e "Health Check: http://localhost:8000/health"
echo ""
echo -e "${YELLOW}Common commands:${NC}"
echo -e "  View logs:    $COMPOSE_CMD -f docker-compose.prod.yml logs -f"
echo -e "  View status:  $COMPOSE_CMD -f docker-compose.prod.yml ps"
echo -e "  Restart:      $COMPOSE_CMD -f docker-compose.prod.yml restart"
echo -e "  Stop:         $COMPOSE_CMD -f docker-compose.prod.yml down"
echo -e "  Full cleanup: $COMPOSE_CMD -f docker-compose.prod.yml down -v"
