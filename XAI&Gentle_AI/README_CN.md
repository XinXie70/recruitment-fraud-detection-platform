# XAI 与 Gentle AI 代码交付包

本目录只包含 XAI、Gentle AI、对应前端展示组件及测试，不包含模型权重、
数据库、虚拟环境、`node_modules` 或其他组员负责的完整前后端代码。

## 目录

```text
backend/
  xai_gentle/
    __init__.py
    contracts.py
    xai_service.py
    gentle_ai_service.py
    gentle_fallback.py
    knowledge/education_en.json
    README.md
  tests/
    test_xai_service.py
    test_gentle_ai.py

frontend/src/features/
  xai_gentle/
    AttributionTable.jsx
    ExplanationText.jsx
    GentleGuidance.jsx
    ModelContributions.jsx
    attributionFormatting.js
  analysis/
    AttributionTable.test.jsx
    ExplanationText.test.jsx
    ModelContributions.test.jsx
```

## 后端功能

- `xai_service.py`：使用 SHAP Partition 解释正式 ensemble 风险评分；长文本使用分层 SHAP。
- `contracts.py`：定义 XAI 和 Gentle AI 的稳定输入输出结构。
- `gentle_ai_service.py`：根据 ensemble 风险和结构化 XAI 证据生成安全说明，可选调用本地 Ollama。
- `gentle_fallback.py`：Ollama 不可用时返回确定性的模板内容。
- `knowledge/education_en.json`：本地教育知识库。

后端通过以下公共接口接入：

```python
from xai_gentle import GentleAIService, RiskContext, XAIService

xai_result = XAIService().explain(text, score_batch, risk_score)
gentle_result = GentleAIService().generate(risk_context, xai_result)
```

正式项目仍需要在 `backend/services/analysis_service.py` 中提供：

- 原始招聘文本 `text`；
- 正式 ensemble 的批量评分函数 `score_batch`；
- 正式 ensemble 最终风险分数 `risk_score`。

后端依赖必须包含：

```text
shap>=0.46.0,<1.0.0
```

## 前端功能

- `ExplanationText.jsx`：根据后端 `start/end` 在原文中高亮证据。
- `AttributionTable.jsx`：显示短语、风险方向和 SHAP 贡献值。
- `ModelContributions.jsx`：显示 BERT 主模型和 LR false-positive gate 的决策角色。
- `GentleGuidance.jsx`：显示 Gentle AI 摘要、安全建议和免责声明。
- `attributionFormatting.js`：格式化 SHAP 百分点。

正式项目的 `frontend/src/App.jsx` 需要导入这些组件：

```javascript
import AttributionTable from './features/xai_gentle/AttributionTable';
import ExplanationText from './features/xai_gentle/ExplanationText';
import GentleGuidance from './features/xai_gentle/GentleGuidance';
import ModelContributions from './features/xai_gentle/ModelContributions';
```

## 测试

后端：

```bash
python -m pytest -q backend/tests/test_xai_service.py backend/tests/test_gentle_ai.py
```

前端测试需要放回完整项目后，在 `frontend` 目录执行：

```bash
pnpm test
```

## 注意

该压缩包是模块交付包，不是可以单独启动的网站。团队应把目录按原路径合并进完整项目，
并保留最新 ensemble 的 `score_batch` 接口。XAI 不直接导入或修改 BERT、LR 或其他模型内部代码。
