# GrowthPilot Agent 优化升级方案

> 日期：2026-04-21
> 版本：v1.0
> 基于 ai-app-lab、claude-cookbooks 两个参考项目的 Agent 设计模式分析

---

## 一、优化分析概览

### 1.1 分析方法论

对 growth-pilot-agent 与 ai-app-lab、claude-cookbooks 两个项目进行了深度对比分析，从三个维度识别优化机会：

| 维度 | 分析范围 | 发现问题数 |
|------|----------|-----------|
| Agent 编排与架构 | 编排模式、并行执行、hook 机制、事件流、动态调度 | 4 |
| 提示词与记忆系统 | Prompt 工程、上下文管理、持久记忆、多模型策略、模板系统 | 4 |
| 工具与可观测性 | 工具注册/安全、错误处理、追踪、评估、重试 | 5 |

### 1.2 核心发现

**优势保留**（不需要改的）：
- "工具先行、AI 解读"的设计理念
- LangGraph StateGraph 编排拓扑设计合理
- A/B 实验平台 + 因果推断 + 多触点归因的技术栈
- 多 LLM Provider 支持（OpenAI/DeepSeek/Ollama）
- AgentState TypedDict + reducer 的状态管理

**需优化**（共 13 项，按优先级排列）：

---

## 二、优化项详细设计

### P0 级（高收益、低工作量）

#### OPT-01: 真并行执行

**问题**：`orchestrator.py:83-93` 用 for 循环顺序执行 parallel_agents，非真并行。

```python
# 当前代码 — 伪并行
for agent_name in parallel_agents:
    agent_result = await self._run_agent(agent_name, state)
```

**方案**：使用 `asyncio.gather` 实现真并行。

```python
# 优化后 — 真并行
tasks = [self._run_agent(name, state) for name in parallel_agents]
results = await asyncio.gather(*tasks, return_exceptions=True)
for name, result in zip(parallel_agents, results):
    if isinstance(result, Exception):
        errors.append(f"Agent {name}: {result}")
    else:
        update.update(result)
```

**参考模式**：ai-app-lab 的 `ParallelAgent` + `_merge_agent_run` 使用 `asyncio.wait(FIRST_COMPLETED)` 做并发事件合并。

**预期收益**：整体耗时减少 50-60%（三个并行 agent 耗时从 T1+T2+T3 → max(T1,T2,T3）。

**影响范围**：`src/agents/orchestrator.py` 的 `run()` 方法。

---

#### OPT-02: Prompt 结构化改造

**问题**：所有 agent 的 system prompt 都是简单的职责列表 + "请用 JSON 格式输出"，缺少业务背景注入、推理引导（CoT）、输出结构约束和 few-shot 示例。

**方案**：设计结构化的 prompt 模板，包含 4 个模块：

```
1. [角色定位] — 你是谁，你的专业能力
2. [业务背景] — 动态注入的上下文（季节、KPI基线、用户规模）
3. [推理步骤] — Chain-of-Thought 引导（<analysis> → <strategy> → <output>）
4. [输出格式] — 严格的 JSON Schema 约束 + 示例
```

**参考模式**：
- claude-cookbooks 的角色定位 + 记忆协议 + 具体推理步骤
- ai-app-lab 的 `CustomPromptTemplate` 支持变量注入（时间、位置、历史、系统上下文）

**Prompt 模板示例**（以 ProspectAgent 为例）：

```python
SYSTEM_PROMPT = """\
你是 GrowthPilot 潜客识别专家 Agent。

## 业务背景
你服务于一个日活 5000 万的货运平台，用户主要来自滴滴出行导流。
当前季节：{season}，关键节点：{seasonal_event}

## 你的职责
1. 基于用户行为数据构建特征
2. 预测用户转化意向
3. 对用户进行评分和排序
4. 预测用户生命周期价值 (LTV)
5. 对用户进行分层

## 推理步骤
在输出最终结果前，请先在对应标签中进行推理：
<analysis>
分析数据质量、模型表现、关键发现
</analysis>
<strategy>
基于分析结果制定分层运营策略
</strategy>

## 输出格式
<output>
{
  "summary": "总体概述 (2-3 句话)",
  "confidence": 0.0-1.0,
  "high_value_profile": "高价值用户画像",
  "intent_insight": "转化意向洞察",
  "segment_strategy": "分层运营建议"
}
</output>
"""
```

**预期收益**：分析质量显著提升，减少 JSON 解析失败率，推理过程可追溯。

**影响范围**：
- `src/agents/` 下所有 agent 的 `SYSTEM_PROMPT`
- `src/prompts/templates/` 下的 Jinja2 模板
- `src/core/base.py` 的 `_build_prompt_context()` 方法

---

### P1 级（高收益、中等工作量）

#### OPT-03: Hook 机制

**问题**：`BaseAgent` 没有扩展点，横切关注点（日志、tracing、限流）散落在各 agent 的 try/catch 中。

**方案**：在 `BaseAgent` 中加入 pre/post hook 点。

```python
# src/core/hooks.py
from abc import ABC, abstractmethod
from typing import Any

class PreRunHook(ABC):
    @abstractmethod
    async def on_pre_run(self, agent_name: str, state: dict) -> dict:
        """Agent 执行前，可修改 state"""
        ...

class PostRunHook(ABC):
    @abstractmethod
    async def on_post_run(self, agent_name: str, result: dict, state: dict) -> dict:
        """Agent 执行后，可修改 result"""
        ...
```

**参考模式**：ai-app-lab 的 6 级 hook 系统（PreToolCall/PostToolCall/PreLLM/PostLLM/PreAgent/PostAgent）。

**内置 Hook 实现**：

| Hook | 用途 |
|------|------|
| `TracingHook` | 记录 agent 执行耗时、状态变化 |
| `LoggingHook` | 统一的日志输出格式 |
| `RetryHook` | 自动重试失败的 LLM 调用 |
| `MetricsHook` | 收集执行指标（token 使用、调用次数） |

**影响范围**：`src/core/base.py`、新增 `src/core/hooks.py`、`src/middleware/`。

---

#### OPT-04: 多模型分层策略

**问题**：所有 agent 使用同一个 LLM，无法按任务复杂度分配模型。

**方案**：在 `BaseAgent` 和 `create_llm` 中支持模型分层。

```python
# 模型分层定义
MODEL_TIERS = {
    "fast": {"model": "deepseek-chat", "temperature": 0.3},      # 便宜、快速
    "default": {"model": "gpt-4o-mini", "temperature": 0.5},     # 平衡
    "power": {"model": "gpt-4o", "temperature": 0.7},            # 强推理
}
```

**使用场景**：

| Agent / 任务 | 推荐层级 | 原因 |
|-------------|---------|------|
| JSON 格式化 / 结果解析 | fast | 简单格式化任务 |
| 单 Agent 策略生成 | default | 平衡质量和成本 |
| Orchestrator 综合分析 | power | 需要强推理能力 |
| 跨 Agent 策略推荐 | power | 复杂多维度分析 |

**参考模式**：claude-cookbooks 的 Haiku（提取）+ Sonnet（评估）+ Opus（综合）分层。

**预期收益**：成本降低 40-60%，同时保持关键路径的分析质量。

**影响范围**：`src/core/llm_factory.py`、`src/core/config.py`、`src/agents/` 下各 agent。

---

#### OPT-05: 事件流进度反馈

**问题**：`run_workflow()` 只返回最终结果，中间过程对用户不可见。

**方案**：在 `AgentState` 中增加 `events` 字段，在 SSE 接口中推送进度。

```python
# State 扩展
class AgentState(TypedDict):
    ...
    events: Annotated[list[dict], operator.add]  # 新增

# 节点中发出事件
async def prospect_node(state):
    events = [{"agent": "prospect", "status": "started"}]
    features = self._feature_engine.build_feature_matrix(data)
    events.append({"agent": "prospect", "status": "running", "step": "feature_engine", "progress": 30})
    ...
    return {"prospect_results": result, "events": events}
```

**参考模式**：ai-app-lab 的 `AsyncIterable[BaseEvent]` 事件流，定义了 `MessageEvent`、`ToolCallEvent`、`ToolCompletedEvent` 等事件类型。

**预期收益**：用户体验改善，长耗时任务不再"黑盒"等待。

**影响范围**：`src/core/state.py`、`src/graph/workflow.py`、Web SSE 接口。

---

### P2 级（中等收益、中等工作量）

#### OPT-06: 持久化记忆系统

**问题**：每次 `run_workflow()` 从零开始，无法积累业务洞察。

**方案**：轻量级文件记忆系统。

```python
# src/core/memory.py
class GrowthMemory:
    """跨会话的增长策略记忆"""

    def __init__(self, base_path: str = "./data/memory"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_insight(self, category: str, key: str, insight: str):
        """保存业务洞察"""

    async def recall(self, category: str, context: str, top_k: int = 3) -> list[dict]:
        """召回相关记忆"""

    async def save_strategy_result(self, strategy_id: str, metrics: dict):
        """保存策略执行效果"""
```

**记忆分类**：

| 分类 | 内容示例 |
|------|----------|
| `seasonal` | 季节性模式、节假日策略效果 |
| `strategy` | 历史策略及其效果指标 |
| `cohort` | 各同期群的行为特征 |
| `model` | 模型训练结果和性能指标 |

**参考模式**：claude-cookbooks 的 `MemoryToolHandler`（路径校验 + 文件类型限制 + CRUD 操作）。

**影响范围**：新增 `src/core/memory.py`，修改 `src/core/base.py`、各 agent 的 prompt 构建。

---

#### OPT-07: 工具注册与安全

**问题**：工具硬编码在各 Agent 中，缺少输入校验和安全防护。

**方案**：

1. **工具注册表**（参考 ai-app-lab 的 ToolPool）：

```python
# src/tools/registry.py
class ToolRegistry:
    _tools: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(tool_cls):
            cls._tools[name] = tool_cls
            return tool_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._tools:
            raise ValueError(f"Unknown tool: {name}")
        return cls._tools[name](**kwargs)
```

2. **安全加载器**（参考 claude-cookbooks 的路径校验）：

```python
class SecureDataLoader:
    ALLOWED_EXTENSIONS = {".csv", ".parquet", ".pq"}
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

    def load(self, path: str) -> dict:
        p = Path(path).resolve()
        if p.suffix not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"不支持的文件类型: {p.suffix}")
        if p.stat().st_size > self.MAX_FILE_SIZE:
            raise ValueError("文件过大")
        # 安全加载...
```

**影响范围**：新增 `src/tools/registry.py`，修改 `src/tools/common/data_loader.py`，各 agent 的 `__init__`。

---

#### OPT-08: 重试与优雅降级

**问题**：错误处理只有 try/catch + logger.warning，没有重试机制。

**方案**：

1. **LLM 调用重试**（使用 tenacity）：

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class BaseAgent:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, RateLimitError)),
    )
    async def _invoke_llm(self, message: str) -> str:
        ...
```

2. **工具降级**：

```python
async def _run_tool_with_fallback(self, tool, primary_args, fallback_args=None):
    try:
        return tool.run(**primary_args)
    except Exception as exc:
        logger.warning(f"Tool {tool.__class__.__name__} failed: {exc}")
        if fallback_args is not None:
            return tool.run(**fallback_args)
        return {"error": str(exc), "fallback": True}
```

**参考模式**：
- ai-app-lab 的 APIException 层级 + 速率限制自动重试
- claude-cookbooks 的 graceful degradation 模式

**影响范围**：`src/core/base.py`、`pyproject.toml`（添加 tenacity 依赖）。

---

### P3 级（锦上添花）

#### OPT-09: 动态 Handoff 机制

**问题**：编排器用关键词匹配做意图检测，Agent 之间不能动态切换。

**方案**：参考 ai-app-lab 的 `build_handoff()` 模式，让 LLM 决定何时切换到哪个 agent。

**影响范围**：`src/agents/orchestrator.py`、`src/core/base.py`。

---

#### OPT-10: 执行追踪

**问题**：只有基础 logging，无法回溯执行路径。

**方案**：轻量级 tracing 装饰器。

```python
@asynccontextmanager
async def trace_agent(name: str, state: AgentState):
    start = time.time()
    trace_id = state.get("metadata", [{}])[-1].get("trace_id", "unknown")
    yield
    elapsed = time.time() - start
    # 写入 state metadata
```

**影响范围**：`src/graph/workflow.py` 各节点函数。

---

#### OPT-11: Agent 质量自评

**问题**：缺少对 Agent 输出质量的评估机制。

**方案**：在 `BaseAgent` 中增加可选的自评步骤，使用 fast 模型评估输出质量。

```python
async def _self_evaluate(self, result: dict) -> float:
    prompt = f"评估以下分析结果的完整性和实用性 (0-1): ..."
    score = await self._invoke_llm(prompt, model_tier="fast")
    return float(score)
```

**参考模式**：claude-cookbooks 的 Promptfoo 评估框架。

**影响范围**：`src/core/base.py`。

---

#### OPT-12: Prompt 模板系统激活

**问题**：`src/prompts/templates/` 目录存在但模板为空，所有 prompt 用 f-string 硬编码。

**方案**：启用 Jinja2 模板，注入动态上下文（当前日期/季节、KPI 基线、上次分析结果）。

**参考模式**：ai-app-lab 的 `CustomPromptTemplate`（支持变量注入：时间、位置、历史对话、系统上下文）。

**影响范围**：`src/prompts/`、各 agent 的 prompt 构建方法。

---

## 三、实施路线图

### Phase 1 — 基础架构加固（P0）

| 编号 | 优化项 | 预估工作量 |
|------|--------|-----------|
| OPT-01 | 真并行执行 | 1h |
| OPT-02 | Prompt 结构化改造 | 3h |

### Phase 2 — 工程化增强（P1）

| 编号 | 优化项 | 预估工作量 |
|------|--------|-----------|
| OPT-03 | Hook 机制 | 2h |
| OPT-04 | 多模型分层 | 2h |
| OPT-05 | 事件流反馈 | 2h |

### Phase 3 — 智能化升级（P2）

| 编号 | 优化项 | 预估工作量 |
|------|--------|-----------|
| OPT-06 | 持久化记忆 | 3h |
| OPT-07 | 工具注册与安全 | 2h |
| OPT-08 | 重试与降级 | 1h |

### Phase 4 — 锦上添花（P3）

| 编号 | 优化项 | 预估工作量 |
|------|--------|-----------|
| OPT-09 | 动态 Handoff | 3h |
| OPT-10 | 执行追踪 | 1h |
| OPT-11 | 质量自评 | 1h |
| OPT-12 | 模板系统激活 | 2h |

---

## 四、参考模式来源索引

| 参考项目 | 关键模式 | 对应优化项 |
|---------|---------|-----------|
| ai-app-lab | `ParallelAgent` + `_merge_agent_run` | OPT-01 |
| ai-app-lab | `CustomPromptTemplate` + 上下文注入 | OPT-02, OPT-12 |
| ai-app-lab | 6 级 Hook 系统 | OPT-03 |
| claude-cookbooks | Haiku/Sonnet/Opus 分层策略 | OPT-04 |
| ai-app-lab | `AsyncIterable[BaseEvent]` 事件流 | OPT-05 |
| claude-cookbooks | `MemoryToolHandler` 持久记忆 | OPT-06 |
| ai-app-lab | `ToolPool` 注册 + claude-cookbooks 安全校验 | OPT-07 |
| ai-app-lab + claude-cookbooks | APIException + 重试 + 降级 | OPT-08 |
| ai-app-lab | `build_handoff()` 动态调度 | OPT-09 |
| ai-app-lab | telemetry + tracing | OPT-10 |
| claude-cookbooks | Promptfoo 评估框架 | OPT-11 |
