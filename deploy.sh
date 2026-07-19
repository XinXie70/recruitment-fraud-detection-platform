#!/usr/bin/env bash
# ============================================================
# 生产环境一键部署脚本 — 后端 + 前端 + 数据库
# （模型推理已拆分到独立服务器，本脚本不部署模型服务）
#
# 用法: chmod +x deploy.sh && ./deploy.sh
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Fake Job Detection — 后端+前端+数据库部署${NC}"
echo -e "${GREEN}============================================${NC}"

# 1. 检查 .env 文件
if [ ! -f .env ]; then
    if [ -f .env.prod ]; then
        echo -e "${YELLOW}[!] .env 不存在，从 .env.prod 复制...${NC}"
        cp .env.prod .env
        echo -e "${RED}[⚠] 请先编辑 .env 文件，修改数据库密码和 SECRET_KEY！${NC}"
        echo -e "${RED}    如果使用远程模型服务器，还需设置 MODEL_SERVER_URL${NC}"
        echo -e "${RED}    编辑完成后重新运行: ./deploy.sh${NC}"
        exit 1
    else
        echo -e "${RED}[✗] .env.prod 也不存在，请先创建 .env 文件${NC}"
        exit 1
    fi
fi

# 2. 检查 Docker
if ! command -v docker &>/dev/null; then
    echo -e "${RED}[✗] 未找到 Docker，请先安装 Docker${NC}"
    exit 1
fi

if ! docker compose version &>/dev/null && ! docker-compose version &>/dev/null; then
    echo -e "${RED}[✗] 未找到 Docker Compose，请先安装${NC}"
    exit 1
fi

# 3. 确定 compose 命令
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

echo -e "${GREEN}[3/5] 构建前端镜像...${NC}"
$COMPOSE_CMD -f docker-compose.prod.yml build frontend

# 7. 启动服务
echo -e "${GREEN}[4/5] 启动所有服务...${NC}"
$COMPOSE_CMD -f docker-compose.prod.yml up -d

# 8. 等待健康检查
echo -e "${GREEN}[5/5] 等待服务就绪...${NC}"
sleep 5

# 显示状态
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  部署完成！${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "前端页面:    http://localhost"
echo -e "后端 API:    http://localhost:8000"
echo -e "API 文档:    http://localhost:8000/docs"
echo -e "健康检查:    http://localhost:8000/health"
echo ""
echo -e "${YELLOW}常用命令:${NC}"
echo -e "  查看日志:  $COMPOSE_CMD -f docker-compose.prod.yml logs -f"
echo -e "  查看状态:  $COMPOSE_CMD -f docker-compose.prod.yml ps"
echo -e "  重启服务:  $COMPOSE_CMD -f docker-compose.prod.yml restart"
echo -e "  停止服务:  $COMPOSE_CMD -f docker-compose.prod.yml down"
echo -e "  完全清理:  $COMPOSE_CMD -f docker-compose.prod.yml down -v"
