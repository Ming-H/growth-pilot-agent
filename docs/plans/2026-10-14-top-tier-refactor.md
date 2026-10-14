# Top-Tier Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让项目宣称的每个机制都真实生效（hooks/记忆/数据路径/模型分级/容灾/评估），修复全部已知缺陷，测试全绿，文档与现实一致 —— 达到"可以放心给面试官逐行讲"的工程质量。

**Architecture:** 保持 plan→execute→evaluate→refine/approval→report 的 StateGraph DAG 不变，把断裂的接线补上：execute_node 改走 `BaseAgent.run()` 完整生命周期；熔断器按 (provider, model) 进程级共享；fallback 异构化；CostTracker 按运行归因；专家结果标准化 `kpis` 字典；ReportGenerator 接为确定性 fallback；evals 接为 `gpa eval` 离线评估。

**Tech Stack:** LangGraph StateGraph, LangChain, Pydantic v2, FastAPI, uv, pytest, ruff

---

## 执行约定

- 每个任务完成后跑该任务的测试文件；每个 Phase 结束跑全量 `uv run python -m pytest tests/ -q`
- 不自动 git commit（等用户确认）；每个 Phase 结束提供 diff 摘要
- 修改行为先改/补测试再改实现（对有测试覆盖的部分）

## Task 0: 环境重建（P0）

**Files:** 无代码修改

1. `uv sync --group dev` — 重建 venv（修复 `.venv/bin/python3` 丢失）
2. 验证 `uv run python -m pytest tests/ -q --tb=line` 记录基线（预期 2 failed: routing 测试 + cost float 测试）
3. Makefile：`test` 目标改 `uv run python -m pytest tests/ -v --tb=short`；删除误导性的 `test/unit`-only 目标；`lint` 加 `ruff format --check`

## Task 1: CostTracker 按运行归因（P0）

**Files:**
- Modify: `src/core/token_tracker.py`
- Modify: `src/graph/graph.py`（run_analysis 开头 reset）
- Modify: `src/graph/nodes.py` report_node（token_usage 真实 input/output 拆分）
- Test: `tests/test_graph/test_nodes.py`（已有 cost float 断言）

实现：
- `total_cost` → `round(float(sum(...)), 6)`（修 int 0 问题）
- `report()` 增加 `input_tokens`/`output_tokens` 汇总字段
- `run_analysis()` 开头调用 `reset_cost_tracker()`（每次分析独立归因；web 多请求不互相污染）
- report_node 的 `token_usage` 用 `cost_report["input_tokens"]/["output_tokens"]`

## Task 2: 修 routing 测试（P0）

**Files:** Modify `tests/test_graph/test_workflow.py:64`

实现：`_route_after_evaluate` 返回 `"refine"`（实现正确，测试期望是旧拓扑名 `"execute"`）。更新断言与测试名。

## Task 3: Hooks 真实生效 + success 契约（P0，核心）

**Files:**
- Modify: `src/graph/nodes.py` execute_node
- Modify: `src/core/expert.py`（`_execute` 提取 `state["params"]`；analyze 成功路径显式 `success=True`）
- Modify: `src/graph/nodes.py` `_validate_expert_result`（剥离 `_` 前缀键后再验证，验证后回填）

实现：
```python
# execute_node._run_expert
result = await expert.run({"params": params})   # 走完整 hook 生命周期
# ExpertAgentBase._execute
async def _execute(self, state):
    params = state.get("params", state)
    return await self.analyze(params)
# analyze() 成功返回前: results["success"] = not errors
# _validate_expert_result: private = {k:v for k in result if k.startswith("_")}
#                          public 校验后 {**validated, **private}
```

## Task 4: 评估/精炼链路收敛（P1）

**Files:**
- Modify: `src/graph/nodes.py` evaluate_node（跳过 `success=False` 的专家：不调 LLM，直接记 0 分 + failed 标记，不触发 refine）
- Modify: `src/graph/nodes.py` refine_node（包 `asyncio.timeout(settings.expert_timeout_seconds)`）
- Modify: `src/core/evaluator.py`（`create_resilient_llm` 替代 `create_llm`，评估调用也有容灾）

## Task 5: data_path 贯穿（P1）

**Files:**
- Modify: `src/graph/state.py`（`data_path: NotRequired[str]`）
- Modify: `src/graph/graph.py` run_analysis（接收并放入 initial_state）
- Modify: `src/graph/workflow.py`（透传 data_path——当前被静默丢弃）
- Modify: `src/graph/nodes.py` execute_node（params 携带 data_path）

## Task 6: 共享熔断器 + 异构 fallback（P1）

**Files:**
- Modify: `src/core/llm_factory.py`
- Modify: `src/core/config.py`
- Test: `tests/test_core/`（新增共享熔断状态测试）

实现：
```python
# llm_factory.py 模块级注册表
_breaker_registry: dict[tuple[str, str], CircuitBreaker] = {}
_lock = threading.Lock()
def _get_breaker(provider, model, threshold, timeout) -> CircuitBreaker:  # 同 key 共享
# create_llm 的 fallback 分支：fallback_base_url / fallback_api_key 独立配置
# create_resilient_llm: fallback 与 primary 同 provider+model+base_url 时 → 警告并禁用（自我 fallback 无意义）
```
- config 增加 `fallback_api_key: str = ""`、`fallback_base_url: str = ""`；默认 `fallback_provider="deepseek"`（跨厂商真容灾；未配 key 时创建失败→优雅回退 primary-only，行为不变但日志明确）

## Task 7: 模型分级真实路由（P1）

**Files:**
- Modify: `src/graph/nodes.py` report_node（`tier="power"`——综合报告需要最强推理）
- Modify: `src/middleware/__init__.py` RetryMiddleware（指数退避读 `settings.retry_min_wait/retry_max_wait`）
- Modify: `src/core/config.py`（`model_config` 加 `protected_namespaces=("settings_",)` 消除 warning）

## Task 8: 记忆接入主管线（P1）

**Files:**
- Modify: `src/graph/nodes.py` plan_node（按 org_id 检索 MemoryManager，命中的历史洞察写入 plan.context_summary；异常静默降级——记忆不可用不阻塞分析）
- Modify: `src/graph/nodes.py` execute_node（params 携带 `memory_context`）

## Task 9: KPI 标准化聚合（P1）

**Files:**
- Modify: `src/agents/{prospect,subsidy,ad,retention,conversion}.py`（结果增加 `kpis: dict[str, float]`）
- Modify: `src/graph/workflow.py`（kpi_snapshot = 合并各专家 `kpis`，替代硬编码顶层键——ad_cpa 永远为 0 的 bug 随之修复）
- Modify: `src/core/models.py`（AgentResult 基类加 `kpis: dict[str, float]` 字段）

## Task 10: 合成标签与特征相关（P1，面试诚信）

**Files:**
- Modify: `src/agents/prospect.py`（`synthetic_labels` 改为特征线性组合 + sigmoid + 噪声：`y ~ Bernoulli(sigmoid(w·x))`，w 固定种子生成——AUC 有信息量且可复现）
- Test: 验证合成标签与特征相关性（AUC 显著 > 0.5 的 smoke test 可选）

## Task 11: 护栏误报收敛（P2）

**Files:**
- Modify: `src/guardrails/input_guard.py`
- Test: `tests/test_guardrails/`（补正常查询不误伤用例）

实现：`INJECTION_PATTERNS` 收敛为明确攻击短语（`ignore previous instructions`、`disregard your system prompt`、`jailbreak`、`DAN mode` 等），加中文模式（`忽略.*指令|扮演.*系统|越狱`）；删除裸 `pretend`/`act as`/`summarize.*above` 误伤模式。

## Task 12: 死代码处置（P2）

**Files:**
- Delete: `src/graph/expert_subgraph.py`（被 Task 3 的 hooks-in-run 取代；其验证逻辑已在 nodes.py 边界统一）
- Delete: `tests/unit/`（空目录 agents/middleware + 与 tests/test_tools 重复的工具测试迁移确认）
- Delete: `src/core/base.py::_build_prompt_context`（未调用）

## Task 13: ReportGenerator 接活（P2）

**Files:** Modify: `src/graph/nodes.py` report_node

实现：LLM 报告失败时的 fallback 从"简陋字符串"升级为 `ReportGenerator().generate_report(state)`（已兼容新 state 格式，纯确定性）。

## Task 14: evals 接活为 `gpa eval`（P2）

**Files:**
- Modify: `src/cli.py`（新增 `eval` 命令）
- 实现：`gpa eval --router` 离线确定性评估（builtin 数据集 vs `_intent_classify` 路由命中率，无 LLM、CI 可跑）；`gpa eval --full` 跑图 + expected_keys 覆盖率（需 LLM）

## Task 15: 文档与版本（P2）

**Files:**
- Rewrite: `README.md`（当前架构图、真实亮点、honest 的配置说明含 fallback）
- Update: `CLAUDE.md`
- Update: `CHANGELOG.md`、`pyproject.toml`（7.0.0）

## Task 16: 端到端验证（P0 收尾）

1. `uv run python -m pytest tests/ -q` 全绿
2. `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/` 双清
3. 用 .env 真实 key：`uv run gpa analyze --scope subsidy --query "优化补贴预算分配" --budget 50000` 端到端成功出报告
4. `uv run gpa eval --router` 输出命中率
