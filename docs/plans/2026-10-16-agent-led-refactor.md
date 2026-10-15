# Agent-Led Refactor Implementation Plan (v8.0.0)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将编排范式从"workflow 主导（固定 DAG）"升级为"经典 Agent 主导"——LLM 在工具调用循环中动态决定调用哪些专家、什么顺序、结果是否充分、何时产出报告；原 DAG 降级为可切换的备选模式。

**Architecture:** Chief Agent = 原生 tool-calling 循环（`bind_tools` + LangGraph agent⇄tools 节点循环）。5 个专家 + 记忆检索下沉为 Agent 工具（Agent-as-Tool 模式）；专家内部仍走 `BaseAgent.run()` 完整生命周期（hooks/middleware/Pydantic 校验保留）；质量门（LLM-as-judge）不通过时把反馈注入回 Agent（有界轮次）；HITL 审批中断 + checkpoint 保留；报告 = Agent 最终消息（过输出护栏）+ 运行后记忆回写（记忆闭环）。

**Tech Stack:** LangGraph StateGraph（messages channel + add_messages reducer）、LangChain tools（Pydantic args schema）、GLM-4.7 native function calling（OpenAI 兼容端点）

---

## Task 1: State 与配置

**Files:**
- Create: `src/agent/state.py` — `AgentState(AnalysisState)` 增加 `messages: Annotated[list, add_messages]`
- Modify: `src/core/config.py` — `orchestration_mode: Literal["agent", "workflow"] = "agent"`

## Task 2: Agent 工具层

**Files:** Create `src/agent/tools.py`

设计：工具 = **spec**（name/description/args Pydantic schema + handler(state, args)）。
`build_tool_specs()` 返回：

| 工具 | 参数 | 行为 |
|---|---|---|
| `run_prospect_analysis` | query | 调 ProspectExpert（hooks 全生命周期）→ Pydantic 校验 → 完整结果写入 state（reducer）→ 返回**精简摘要**（kpis + analysis.summary，防爆 context） |
| `run_conversion_analysis` | query | 同上 |
| `run_subsidy_analysis` | query, budget | 同上（budget 必填） |
| `run_retention_analysis` | query | 同上 |
| `run_ad_analysis` | query | 同上 |
| `search_memory` | query | MemoryManager.search（org 隔离），返回命中条目摘要 |

`to_langchain_tools(specs)` → 供 `bind_tools` 的 schema 列表；`execute_tool_call(state, name, args)` → 节点侧分发执行。

## Task 3: Agent 节点与路由

**Files:** Create `src/agent/chief.py`

- `chief_agent_node(state)`: power 档 LLM + bind_tools；system prompt 定义角色/工具使用规则/预算约束/何时完成；输出 AIMessage append 到 messages
- `agent_tools_node(state)`: 执行最新 AIMessage 的全部 tool_calls（并行）；ToolMessage(s) append；专家工具带 asyncio.timeout
- `should_continue(state)`: 有 tool_calls → "tools"；无 → "evaluate"（质量门）；轮次超限 → 直接 "approval/report"
- `evaluate_gate_node(state)`: 复用 `evaluate_expert_output` 评 Agent 最终回答；低于阈值且 round < max → 反馈 HumanMessage 注入 → 回 agent；否则 → approval
- `agent_report_node(state)`: final_report = Agent 最后 AIMessage（输出护栏过滤）+ 记忆回写 + token/cost 归因

## Task 4: Agent 图

**Files:** Create `src/agent/graph.py`

```
guard(输入护栏) → agent ⇄ tools → evaluate_gate ─┬→ agent（反馈，≤max 轮）
                                                └→ approval(interrupt) → report → END
```

`build_agent_graph()` + `build_compiled_agent_graph()`（checkpointer + interrupt_before=["approval"]）+ `run_agent_analysis(...)` 入口（与 run_analysis 同签名 + 返回兼容 dict）。

## Task 5: 模式分流与 CLI

**Files:**
- Modify: `src/graph/workflow.py` — 按 `settings.orchestration_mode` 分流到 agent/workflow，返回结构保持兼容（expert_results/kpi_snapshot 聚合逻辑共用）
- Modify: `src/cli.py` — `analyze --mode agent|workflow`（默认取配置）；info 显示当前模式

## Task 6: 测试

**Files:** Create `tests/test_agent/test_tools.py`、`test_agent_graph.py`

- 工具 spec/schema 生成 + execute_tool_call 分发（mock 专家）
- agent→tools→agent 循环路由（mock AIMessage with/without tool_calls）
- 质量门反馈循环（低分注入反馈、超限直通）
- 端到端：mock LLM 两轮（调一个专家→出报告）→ state 断言
- 旧 workflow 模式回归：mode=workflow 走旧 DAG

## Task 7: 验证 + 文档 + 发布

- 全量测试绿 + ruff 双清
- 真实 GLM key 端到端（agent 模式，观察真实 tool_calls 决策）——**依赖用户提供有效 key**
- README（Agent 主导为主图，workflow 为备选）、CLAUDE.md、CHANGELOG 8.0.0、版本号
- 提交推送（确认 .env 不入库）、CI 绿
