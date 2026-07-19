# 谷歌云部署指南 — 后端（数据库使用 Supabase 免费 PostgreSQL）

> 只部署后端（FastAPI + Gunicorn）到 Cloud Run，数据库使用 Supabase 免费托管 PostgreSQL，前端不部署。

## 📋 架构

```
用户/前端 → Cloud Run (backend) → Supabase (免费 PostgreSQL)
                ↕
         Model Inference Server (已有的，不在这里部署)
```

## 🚀 一次性准备（~15 分钟）

### 1. 创建 Supabase 免费数据库（5 分钟）

1. 打开 [supabase.com](https://supabase.com) 注册/登录
2. 点击 **New project**
3. 填写项目名（如 `almond-db`），设置数据库密码（记下来！）
4. Region 选择 **ap-southeast-1 (Singapore)** 或 **us-west-1**（靠近你的 Cloud Run 区域）
5. 选择 **Free plan**，点击 Create project
6. 等待创建完成（约 2 分钟）
7. 进入 **Settings → Database**，找到 **Connection string**
8. 选择 **URI** 标签，复制连接串，格式如下：

```
postgresql://postgres.[PROJECT_REF]:[YOUR_PASSWORD]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres
```

> ⚠️ **重要**: 使用 **Session Pooler** 连接（端口 6543），而不是直连（端口 5432），因为 Cloud Run 是无服务器的，连接池更合适。

### 2. 安装 gcloud CLI 并登录

```bash
brew install google-cloud-sdk
gcloud init
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 3. 启用必要的 API

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com
```

### 4. 创建 Artifact Registry（存放 Docker 镜像）

```bash
gcloud artifacts repositories create almond-repo \
  --repository-format=docker \
  --location=asia-southeast1
```

### 5. 在 Secret Manager 中存储密钥

```bash
# SECRET_KEY（用 Python 生成一个随机字符串）
python3 -c "import secrets; print(secrets.token_urlsafe(64))" | \
  gcloud secrets create SECRET_KEY_PRODUCTION --data-file=-

# DATABASE_URL — 替换为你的 Supabase 连接串
echo -n "postgresql+psycopg2://postgres.xxxxx:YOUR_PASSWORD@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres" | \
  gcloud secrets create DATABASE_URL_PRODUCTION --data-file=-

# ACCESS_TOKEN_EXPIRE_MINUTES
echo -n "10080" | gcloud secrets create ACCESS_TOKEN_EXPIRE_MINUTES --data-file=-
```

### 6. 授予 Cloud Run 访问 Secret Manager 的权限

```bash
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
CLOUD_RUN_SA="service-${PROJECT_NUMBER}@serverless-robot-prod.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member="serviceAccount:${CLOUD_RUN_SA}" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 🔧 部署

### 方式一：Cloud Build 自动部署（推荐）

修改 `cloudbuild.yaml` 中的 `${_INSTANCE_CONNECTION_NAME}` 和 `${_MODEL_SERVER_URL}` 为你实际的
值，然后推送代码即可触发自动构建部署。

### 方式二：手动构建部署

```bash
# 1. 构建镜像
gcloud builds submit \
  --config=cloudbuild.yaml \
  --substitutions=_MODEL_SERVER_URL="https://your-model-server.com"

# 2. 或者只用 Docker 手动部署
docker build -f backend/Dockerfile.cloudrun -t gcr.io/YOUR_PROJECT/backend .
docker push gcr.io/YOUR_PROJECT/backend

gcloud run deploy almond-backend \
  --image=gcr.io/YOUR_PROJECT/backend \
  --region=asia-southeast1 \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars=ENVIRONMENT=production \
  --set-env-vars=MODEL_SERVER_URL="https://your-model-server.com" \
  --set-secrets=DATABASE_URL=DATABASE_URL_PRODUCTION:latest \
  --set-secrets=SECRET_KEY=SECRET_KEY_PRODUCTION:latest
```

---

## ✅ 验证部署

```bash
# 获取服务 URL
gcloud run services describe almond-backend \
  --region=asia-southeast1 \
  --format='value(status.url)'

# 健康检查
curl https://YOUR_SERVICE_URL/api/health

# 预期返回:
# {"status":"healthy","service":"fake_job_detection_api","model_ready":true}
```

---

## 📊 费用估算

| 服务 | 配置 | 月费（约） |
|------|------|-----------|
| Cloud Run | 0.5 vCPU, 256MB, 按量 | $0 — 免费额度内 |
| Supabase | 免费套餐 (500MB DB) | **$0** |
| Secret Manager | 3 个密钥 | $0 |
| Artifact Registry | 少量镜像 | ~$0 |
| **合计** | | **$0/月 🎉** |

---

## 🔒 安全注意事项

1. **Supabase 连接串包含密码** — 务必通过 Secret Manager 存储，不要硬编码
2. **Secret Manager 存储所有密钥** — 不要在代码或环境变量中硬编码
3. **Cloud Run 自动 HTTPS** — 不需要自己配置 TLS 证书
4. **设置 CORS_ORIGINS** — 限制允许访问的前端域名
5. **Supabase 免费套餐限制** — 500MB 数据库、2 个项目、每周备份、暂停后需手动恢复
