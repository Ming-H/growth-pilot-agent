# 实验：DSPy 优化规划节点（agent_plan_node）

> 把 `src/agent/chief.py` 的手写规划提示词（`PLAN_SUBGOALS_PROMPT`）变成
> DSPy 可编译的 Signature，用任务级基准（τ-bench for Growth）的
> `expected_experts` 真值当训练信号，MIPROv2 自动优化指令——
> 三方对比：生产规则路由（`plan:rule` 兜底）/ 手写提示词 / 编译产物。

姊妹实验：[tally/experiments/dspy_understand](https://github.com/Ming-H/tally)
（同一套方法在 RAG 查询理解节点上的落地）。

## 为什么选规划节点

agent 模式的链路是 `plan → chief ⇄ tools → evaluate → approval → report →
kpi_gate`，规划是第一环：它产出的子目标清单决定调度哪些专家、花多少预算。
规划错了后面全错——而它恰好是**纯结构化输出**任务，真值现成
（`builtin_tasks()` 每题带 `expected_experts`），单次评估只要一次 LLM
调用，是 MIPROv2 理想的优化对象。

## 设计

### 候选通道：与线上同一道闸

DSPy 模块的输出先经**生产校验器** `_validate_subgoals` 清洗（白名单交集、
按专家去重、封顶 5 个、只删不造），再进入评分——候选与线上走的完全是
同一道闸，优化不会"考过闸门"。

### 评分（`metric.py`，纯函数，CI 可测）

```
真值为空（honesty 拒答题）：候选为空 = 1.0，非空 = 0.0
精确匹配真值            = 1.0
超集（全覆盖 + 多选）    = 0.75   ← 路由对但浪费预算
部分覆盖                = 0.5 × 召回
全漏                    = 0
```

### 对照组

| 基线 | 是什么 | 成本 |
|---|---|---|
| 规则路由 | `_intent_classify`（`plan:rule` 兜底同款） | 零 LLM |
| 优化前 | 未编译 `dspy.Predict`（手写提示词语义等价物） | 1 次调用/题 |
| 编译产物 | MIPROv2 优化后的指令 | 1 次调用/题 |

## 用法

```bash
# dspy 已入 dev 依赖组（uv add "dspy[optuna]" --group dev）
uv run python experiments/dspy_planner/runner.py             # 冒烟（2 候选×2 轮）
uv run python experiments/dspy_planner/runner.py --full      # 完整 MIPROv2
uv run python experiments/dspy_planner/runner.py --eval-compiled

uv run pytest tests/test_dspy_planner_metric.py -q           # 评分层单测（无 LLM）
```

产物：`compiled/planner.json`（DSPy 程序）+ `report-smoke.json` /
`report-full.json`（三方逐题对比）。

## dspy 3.5 版本事实（改 runner 前先读）

| 坑 | 修法 |
|---|---|
| 默认走 OpenAI Responses API，兼容端点 404 | `dspy.LM(..., model_type="chat")` |
| `auto` 默认非 None，手动预算被拒 | 显式 `auto=None` 再传 num_candidates |
| MIPROv2 贝叶斯优化缺可选依赖 | dev 组装 `dspy[optuna]` |
| Example 输入字段名 | `with_inputs()` 后取 `example.question` |
| `program.save()` | 目录先 mkdir，路径以 `.json` 结尾 |

（与 tally 实验的避坑清单同源，两仓保持同步。）

## 预期与判读

- 规则路由基线在 7 题上本就强劲（关键词路由是 workflow 模式的信任底座），
  优化的价值空间在：**语义变体问法**（规则漏检）、**多专家组合题**、
  **honesty 拒答**（规则不会拒绝超纲问题）。
- 冒烟档验证链路；完整档看 `report-full.json` 的逐题 diff——若编译产物在
  honesty 题上从规则基线的 0 分提到 1.0，即"拒答能力可被优化器教会"的直接
  证据。
