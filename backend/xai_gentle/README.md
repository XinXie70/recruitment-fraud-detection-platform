# XAI 与 Gentle AI 独立模块

这个目录是 XAI/Gentle AI 负责人的独立代码范围。其他组员应从
`xai_gentle/__init__.py` 使用公开接口，不要直接依赖内部文件。

```python
from backend.xai_gentle import GentleAIService, RiskContext, XAIService
```

## 文件

- `contracts.py`：你的稳定输入输出，包括 `RiskContext`、`XAIResult` 和
  `GentleAIResult`。
- `xai_service.py`：解释正式 ensemble scorer，生成 SHAP 或 occlusion 证据。
- `gentle_ai_service.py`：读取结构化证据，可选调用本地 Ollama 改写。
- `gentle_fallback.py`：Ollama 不可用时使用的确定性模板。
- `knowledge/education_en.json`：本地教育知识库。

## 与其他组员的边界

XAI 的公开调用：

```python
XAIService.explain(text, score_batch, expected_output) -> XAIResult
```

它不 import 任何具体模型或 EnsemblePredictor，只调用后端提供的
`score_batch`。

Gentle AI 的公开调用：

```python
GentleAIService.generate(risk, xai) -> GentleAIResult
```

`risk` 是只有四个字段的 `RiskContext`。Gentle AI 不接收模型、原始 scorer
或完整 EnsembleResult，因此不能重新判断真假。

唯一后端集成点是 `backend/services/analysis_service.py`。你的独立测试命令：

```bash
python -m pytest -q tests/xai_gentle
```
