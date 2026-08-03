# Risk Score 与 Risk Level 定义

本文说明本实验对外输出的 **风险分数** 与 **风险档位** 如何定义。  
核心原则：**分数保留模型原始证据，档位体现决策策略（含 FP-gate）**。两者职责分离，避免改写概率造成难以解释的跳变。

相关实验目录：`ensemble_bert_fp_gate_lr_none_bigram_maxlen512`  
Validation 选定参数：

| 参数 | 值 | 含义 |
|---|---:|---|
| `bert_threshold` | 0.30 | BERT 判欺诈 / 进入「高风险」候选的分数线 |
| `lr_gate` | 0.06 | LR 否决门控：仅当 LR 分数低于此值时触发 |
| `t_low`（建议默认） | 0.10 | 「无风险」与「低风险」分界 |

`bert_threshold` 与 `lr_gate` 在 Validation 上按 Fraud F1 网格搜索得到，Test 仅做冻结评估。

---

## 1. Risk Score（0–1）

```text
risk_score = bert_score
```

- **定义**：BERT 输出的原始欺诈概率（fraud class probability）。
- **取值范围**：`[0, 1]`，越接近 1 表示模型越倾向判定为欺诈。
- **重要约定**：
  - Risk score **始终等于** `bert_score`。
  - **不**使用 `min(bert_score, lr_score)` 等方式改写分数。
  - FP-gate **不修改** risk score，只影响 risk level / 最终处置动作。

这样分数始终可解释为「主模型（BERT）给出的风险证据」，便于复核、排序与校准分析。

---

## 2. Risk Level（无 / 低 / 高）

Risk level 由 **score 阈值** + **FP-gate 规则** 共同决定。

### 2.1 先按分数得到候选档位

| 候选档位 | 条件 |
|---|---|
| 无风险（none） | `risk_score < 0.10` |
| 低风险（low） | `0.10 ≤ risk_score < 0.30` |
| 高风险（high） | `risk_score ≥ 0.30` |

其中 `0.30` 与 Validation 选定的 `bert_threshold` 对齐；`0.10` 为建议的业务默认低档分界，可在 Validation 上按「无风险欺诈率上限 / 人工复核容量」再校准。

### 2.2 再应用 FP-gate（只降档，不改分）

```text
gated = (risk_score >= 0.30) and (lr_score < 0.06)

if gated:
    risk_level = 低风险      # 从「高风险」降为「低风险」
else:
    risk_level = 候选档位     # 保持 2.1 的结果
```

含义：

- 默认信任 BERT 分数分档。
- 仅当「BERT 已达高风险线，但 LR 极低（`< 0.06`）」时，认为存在较高误报风险，**将档位从高风险降为低风险**（进入复核，而非直接当无风险清零）。
- `risk_score` 仍显示原来的 BERT 高分，并建议同时输出 `gated_flip=true` 便于解释。

### 2.3 伪代码（完整）

```text
risk_score = bert_score
lr_score   = logistic_regression_fraud_probability

# 候选档位（只看 BERT 分）
if risk_score < 0.10:
    candidate = 无风险
elif risk_score < 0.30:
    candidate = 低风险
else:
    candidate = 高风险

# FP-gate：只降档
gated = (risk_score >= 0.30) and (lr_score < 0.06)
if gated:
    risk_level = 低风险
else:
    risk_level = candidate
```

---

## 3. 与二分类决策的关系

若系统仍需 0/1 预测：

| 输出 | 建议定义 |
|---|---|
| `prediction = 1`（欺诈/拦截） | `risk_level == 高风险` |
| `prediction = 0` | `risk_level ∈ {无风险, 低风险}` |

这与 FP-gate 实验中的硬决策一致：被门控样本不再作为高风险拦截，但分数仍保留 BERT 原值。

---

## 4. 推荐对外字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `risk_score` | float `[0,1]` | `= bert_score`，模型原始欺诈概率 |
| `risk_level` | `{无风险, 低风险, 高风险}` | 由阈值 + FP-gate 决定 |
| `bert_score` | float `[0,1]` | 与 `risk_score` 相同，可冗余保留 |
| `lr_score` | float `[0,1]` | LR 欺诈概率，用于门控与解释 |
| `gated_flip` | bool | 是否触发 FP-gate 降档 |
| `prediction` | `{0,1}` | 可选；建议与「高风险」对齐 |

---

## 5. 设计理由（简述）

1. **有说服力**：连续分来自主模型概率，不是临时拼出来的合成分。  
2. **可解释**：出现「分数较高但档位为低风险」时，可用 `gated_flip` + `lr_score < 0.06` 说明是规则降档。  
3. **避免怪跳变**：不再把 0.9 改写成 0.02 这类难以对外解释的分数。  
4. **与实验结果一致**：FP-gate 在 Validation/Test 上通过降误报提升了 Fraud F1；对外输出时把该收益体现在 **level**，而不是改写 **score**。

---

## 6. 业务动作建议（可选）

| Risk level | 建议动作 |
|---|---|
| 无风险 | 自动放行 |
| 低风险 | 人工复核（含被 FP-gate 降档的样本） |
| 高风险 | 拦截或重点处理 |

---

## 7. 相关产物

- 门控选参与指标：`results/config.json`、`results/RESULTS.md`
- 门控搜索脚本：`code/run_fp_gate_ensemble.py`
- 说明：早期结果里的 `ranking_score=min(bert, lr)` 仅用于内部排序试验；**对外正式口径以本文为准**（`risk_score = bert_score`）。
