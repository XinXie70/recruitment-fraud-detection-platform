# Fraud Detection API — Docker

本目录集中存放云端 / 本地容器部署文件。应用代码在 `../api_flask/`。

## 目录

```text
docker/
  Dockerfile                 # CPU 镜像（推荐云端）
  Dockerfile.gpu             # GPU 镜像（可选）
  docker-compose.yml
  entrypoint.sh
  requirements-docker.txt
  README.md
```

构建上下文是项目根目录（`capstone-project-.../`），因此根目录的 `.dockerignore` 会生效。

## 快速启动

```powershell
cd F:\FinalEsemble\capstone-project-26t2-9900-h09c-almond\docker
docker compose up --build -d
```

或手动构建：

```powershell
cd F:\FinalEsemble\capstone-project-26t2-9900-h09c-almond
docker build -f docker/Dockerfile -t fraud-detection-api:latest .
docker run -d --name fraud-detection-api -p 5000:5000 --restart unless-stopped fraud-detection-api:latest
```

- API：`http://127.0.0.1:5000`
- Swagger：`http://127.0.0.1:5000/apidocs/`
- 健康检查：`GET /health`

首次启动会预加载 LR + BERT，约需 1–2 分钟。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | `5000` | 容器监听端口 |
| `HOST` | `0.0.0.0` | 绑定地址 |
| `ALLOW_CPU` | `1` | 无 GPU 时允许 CPU 推理 |
| `GUNICORN_WORKERS` | `1` | 建议保持 1 |
| `GUNICORN_THREADS` | `4` | 并发线程 |
| `GUNICORN_TIMEOUT` | `180` | 请求超时（秒） |

资源建议：CPU 实例 **≥ 2 vCPU / 4GB RAM**。

## 云端部署

```powershell
docker tag fraud-detection-api:latest <registry>/fraud-detection-api:latest
docker push <registry>/fraud-detection-api:latest
```

在 ECS / ACI / Cloud Run / Container Apps 等平台使用该镜像，暴露端口 `5000`（或平台注入的 `$PORT`），健康检查设为 `GET /health`，内存建议 ≥ 4Gi。

## GPU 镜像（可选）

```powershell
cd F:\FinalEsemble\capstone-project-26t2-9900-h09c-almond
docker build -f docker/Dockerfile.gpu -t fraud-detection-api:gpu .
docker run --gpus all -d -p 5000:5000 -e ALLOW_CPU=0 fraud-detection-api:gpu
```
