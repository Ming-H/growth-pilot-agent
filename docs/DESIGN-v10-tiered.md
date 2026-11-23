# DESIGN-v10: 分层 Agent 架构（Tiered Autonomy）

> 核心主张：不是所有查询都配用完整 Agent。
> 按 query 复杂度分层调度——确定性管线兜底、专家 Agent 做中坚、深度 Agent 做疑难。
> 参考：Anthropic multi-agent research system (June 2025) · LangChain multi-agent patterns (Jan 2026)

## 1. 动机

Anthropic 的多 Agent 研究系统实测：多 Agent 比单 Agent 高 90.2%，但 **token 消耗 ~15×**。
多 Agent 的价值边界很清楚（原文）：

- ✅ 适合：breadth-first 查询 + 信息超单上下文窗口 + 多工具交互 + 高价值任务
- ❌ 不适合：需要全 Agent 共享上下文的场景 + Agent 间有大量依赖 + 简单查询

GrowthPilot 的增长分析场景恰好**横跨这两个区间**：
- 简单事实查询（"AUC 多少"）→ 确定性管线足够
- 中等分析（"补贴 ROI 分析"）→ 专家 + 一次 LLM 综合
- 复杂跨域优化（"全局预算分配"）→ 需要 Agent 自主迭代

**结论**：不应该全场景 Agent 化，而应该**分层**——让系统智能判断"这个问题配不配用完整 Agent"。

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                     Supervisor（编排层）                       │
│                                                             │
│  查询理解 → 复杂度判定 → 分派到对应层级 → 结果汇聚 → KPI 闸门    │
│                                                             │
│  复杂度判定 = 确定性规则（硬门槛）+ LLM 判定（灰色地带）         │
├──────────┬──────────────────────┬────────────────────────────┤
│  Tier 0  │       Tier 1         │         Tier 2             │
│ 确定性管线 │     专家 Agent        │      深度分析 Agent          │
│          │                      │                            │
│ 关键词路由 │ 领域Agent + 工具集    │ 领域Agent + ReAct 推理循环   │
│ 固定 DAG  │ 单次分析             │ 多步迭代                    │
│ 模板报告  │ 1-2 次 LLM 调用      │ 3-10 次 LLM 调用            │
│ 零 LLM   │ 步数 ≤ 3             │ 步数 ≤ 10                  │
│          │                      │                            │
│ ~0 成本   │ ~1× 成本             │ ~5× 成本                   │
│ 毫秒级    │ 秒级                 │ 十秒-分钟级                  │
├──────────┴──────────────────────┴────────────────────────────┤
│              全部经过同一个 KPI 对账闸门                        │
│              全部降级可见（state["degradations"]）               │
│              全部可零 LLM 兜底（确定性孪生）                     │
└─────────────────────────────────────────────────────────────┘
```

## 3. 三层定义

### Tier 0 — 确定性管线

现有 workflow 模式（`src/graph/graph.py`）：
- `_intent_classify` 关键词路由 → 固定 DAG → 模板报告
- 零 LLM、零 token、CI 可复现（字节级断言）
- **触发条件**：预算 < 阈值 / 单领域事实查询（有缓存）/ CI 回归 / 监控告警

### Tier 1 — 专家 Agent

现有 agent 模式（`src/agent/graph.py`）+ 微调：
- Chief Agent 调度专家，每个专家 = 确定性管线 + 一次 LLM 综合
- 专家有独立工具集和系统提示
- **新增**：步数硬上限 3（当前无限制，补上护栏）
- **触发条件**：单/双领域分析查询 / 中等预算

### Tier 2 — 深度分析 Agent（新增）

- 每个 Agent 有自己的 **ReAct 推理循环**
- 能根据中间结果迭代（跑完因果推断 → 发现弹性异常 → 换细分段重跑）
- 步数硬上限 10 + 幂等缓存 + 整批重复收手
- 中间结果持久化到 L1 事实台账（已有基础设施）
- **触发条件**：跨 ≥3 领域 / 高预算 / 明确的迭代型分析需求

## 4. Supervisor 设计

### 4.1 复杂度判定（混合策略）

```python
def determine_tier(query: str, budget: float, context: dict) -> Tier:
    """确定性规则优先，LLM 判定灰色地带。"""

    # 硬规则 1: 预算不足 → Tier 0（不烧 token）
    if budget is not None and budget < tier0_budget_threshold:
        return Tier.WORKFLOW

    # 硬规则 2: CI/回归/监控 → Tier 0
    if context.get("mode") == "regression":
        return Tier.WORKFLOW

    # 硬规则 3: 显式指定 → 尊重调用方
    if context.get("force_tier") is not None:
        return context["force_tier"]

    # 硬规则 4: 跨域 + 高预算 → Tier 2
    domains = _count_domains(query)
    if domains >= 3 and (budget or 0) >= tier2_budget_threshold:
        return Tier.DEEP

    # 硬规则 5: 单领域事实查询 → Tier 0
    if domains == 1 and _is_factoid(query):
        return Tier.WORKFLOW

    # 灰色地带 → LLM 快速判定（role=fast, 单次调用）
    return _llm_classify(query, budget)
```

### 4.2 LLM 判定的灰色地带

```
输入: query + budget + domain_count
输出: {"tier": 0|1|2, "reason": "一句话"}

提示词核心约束:
- 简单事实查询打 0（"XX 的 AUC 是多少"）
- 单领域分析打 1（"帮我优化补贴策略"）
- 跨域迭代分析打 2（"对比六种策略并给出全局预算分配"）
- 拿不准打 1（默认中间档，性价比最优）
```

### 4.3 降级链

```
Tier 2 深度 Agent 失败/超时/弱模型
  → 自动降级到 Tier 1（单次分析）
    → LLM 不可用
      → 自动降级到 Tier 0（确定性管线）

每层降级都写 state["degradations"]，报告尾部可见。
零 LLM 时 Supervisor 本身退化为确定性规则（只用硬规则，跳过灰色地带判定）。
```

## 5. 与 v9 的兼容性映射

| v9 组件 | v10 去向 | 变更 |
|---|---|---|
| workflow 模式 (`src/graph/graph.py`) | **Tier 0**（不变） | 无改动 |
| agent 模式 (`src/agent/graph.py`) | **Tier 1**（微调） | 加步数上限 |
| Chief Agent (`src/agent/chief.py`) | **Tier 1 内部**（保留） | 保留 ReAct 循环 |
| 六专家 (`src/agents/*.py`) | **Tier 1 + Tier 2 共用** | Tier 2 时套 ReAct 外壳 |
| KPI 闸门 (`src/core/kpi_gate.py`) | **全层共用**（不变） | 无改动 |
| 三层记忆 (`src/memory/store.py`) | **全层共用**（不变） | 无改动 |
| 工具层 (`src/tools/**`) | **全层共用**（不变） | 无改动 |
| `src/graph/nodes.py` | 保留（Tier 0 的节点实现） | 无改动 |
| **新增** `src/supervisor/` | Supervisor 节点 | 新代码 |
| **新增** `src/agents/deep/` | Tier 2 深度 Agent 外壳 | 新代码 |

**关键设计原则**：Tier 0 和 Tier 1 **完全不改现有代码**——608 个测试零适配。只新增 Supervisor 和 Tier 2 外壳。

## 6. 新增代码结构

```
src/supervisor/
├── __init__.py
├── tier.py         # Tier 枚举 + 判定逻辑（纯函数，可测）
├── node.py         # LangGraph 节点：查询理解 → 复杂度判定 → 路由
└── state.py        # Supervisor 层的 state schema

src/agents/deep/
├── __init__.py
├── base.py         # DeepAgentBase：ReAct 外壳 + 步数上限 + L1 持久化
└── subsidy.py      # SubsidyDeepAgent（PoC）

tests/
├── test_supervisor_tier.py   # 判定逻辑纯函数测试（零 LLM）
└── test_deep_agent.py        # Deep Agent 外壳测试（mock LLM）
```

## 7. 评测适配

| 维度 | Tier 0 | Tier 1 | Tier 2 |
|---|---|---|---|
| 路由命中率 | ✓（现有 router eval） | ✓ | ✓ |
| 任务级基准 | ✓（现有 tasks eval） | ✓ | ✓ + 迭代质量 |
| KPI 闸门 | ✓（现有 gate） | ✓ | ✓ |
| **层级选择正确性** | 新增 | 新增 | 新增 |
| 零 LLM 回归 | ✓（字节级断言） | — | — |
| Token 成本归因 | — | ✓ | ✓ + 步级 |

新增评测维度：给定一组查询，Supervisor 选择的层级是否与人工标注的"最优层级"一致。这是分层架构的核心质量指标。

## 8. 设计决策记录

### D1: 为什么不用 LangGraph create_react_agent？

现有 Chief Agent 已有完整的 ReAct 循环（步级四元组观测 + 幂等缓存 + 弱模型护栏），且与 LangGraph StateGraph 深度集成。用 `create_react_agent` 会丢失这些工程韧性。选择**扩展现有 Chief 循环**，不替换。

### D2: 为什么 Supervisor 的灰色地带判定用 role=fast？

- 确定性规则（硬门槛）覆盖 >70% 的场景，不需要 LLM
- 灰色地带的判定本质上是"分类"任务（3 选 1），fast 模型足够
- 延迟敏感：判定应该在 <2s 内完成，不阻塞主分析流程

### D3: 为什么 Tier 2 步数上限是 10 而不是更多？

Anthropic 实测：3-10 次工具调用适合"直接比较"类任务，>10 次适合"复杂研究"。增长分析更接近前者。且步数越多，误差累积越严重（eₖ·gₖ 传播公式）。

### D4: 为什么保留双模式（agent/workflow）而不是统一为分层？

分层是**调度策略**，双模式是**执行模式**。两者正交：
- Tier 0 = workflow 模式（不变）
- Tier 1 ≈ agent 模式（微调）
- Tier 2 = agent 模式 + 深度外壳（新增）

统一为分层后，`--mode` 参数变成 `--tier`，但底层实现不变。

## 9. 已知边界

1. **Tier 2 的跨 Agent 实时通信**：当前设计是同步的（等所有 Tier 2 Agent 完成再汇总），不做实时流式协作——Anthropic 也明确说这是未来方向
2. **层级判定的准确率**：灰色地带的 LLM 判定可能出错——靠降级链兜底（判高了自动降，判低了损失可控）
3. **Tier 2 Agent 的中间结果质量**：每步结果仍走 KPI 闸门——数字对不上就降级到确定性输出

## 10. Roadmap

| Phase | 内容 | 预估 | 验收标准 |
|---|---|---|---|
| 1 | Supervisor 节点 + 判定逻辑 + 测试 | 1 天 | 12+ 纯函数测试全过 |
| 2 | SubsidyDeepAgent PoC + 测试 | 1 天 | 3+ mock 测试全过 |
| 3 | 全量回归（608 现有测试零适配通过） | 半天 | `make test` 全绿 |
| 4 | README 更新 + commit + push | 半天 | GitHub 可见 |
