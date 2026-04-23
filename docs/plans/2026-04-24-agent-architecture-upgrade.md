# Growth Pilot Agent 架构升级计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将当前 ReAct 循环架构升级为 LangGraph StateGraph DAG 编排，引入 Human-in-the-Loop、Agent Handoff、Vector DB 记忆和完整护栏体系，达到行业生产级标准。

**Architecture:** 以 LangGraph StateGraph 为核心编排引擎，Chief Agent 作为轻量路由器，Expert Agent 独立自治运行，通过 Checkpoint 实现断点恢复和人工审批。记忆系统迁移到 ChromaDB + Embedding 实现语义检索。

**Tech Stack:** LangGraph (StateGraph + Checkpoint), LangChain, ChromaDB, FastAPI, PostgreSQL, OpenTelemetry, Structured Output (Pydantic v2)

---

## Phase 1: LangGraph StateGraph 核心（Tasks 1-3）

### Task 1: 定义 AnalysisState 和 Graph Schema

**Files:**
- Create: `src/graph/state.py`
- Create: `src/graph/nodes.py`（骨架）

**Step 1: 创建 AnalysisState TypedDict**

```python
# src/graph/state.py
from typing import TypedDict, Optional, Annotated
from enum import Enum
import operator

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    AWAITING_APPROVAL = "awaiting_approval"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"

class AnalysisState(TypedDict):
    # 输入
    query: str
    scope: Optional[str]
    budget: Optional[float]
    org_id: str
    user_id: str

    # 规划
    plan: Optional[dict]               # Chief Agent 生成的执行计划
    selected_experts: list[str]        # 需要调用的 Expert 列表

    # 执行
    expert_results: Annotated[list[dict], operator.add]  # 每个 Expert 的结构化结果
    execution_errors: Annotated[list[str], operator.add]  # 执行错误记录

    # 评估
    quality_scores: Optional[dict]     # 评估分数
    needs_refinement: bool             # 是否需要重新执行
    refinement_round: int              # 重试轮次

    # 审批
    approval_required: bool            # 是否需要人工审批
    approved: Optional[bool]           # 审批结果

    # 输出
    final_report: Optional[str]        # 最终报告
    status: AnalysisStatus
    token_usage: dict                  # Token 消耗统计
    cost_usd: float                    # 成本统计
```

**Step 2: 创建节点骨架文件**

```python
# src/graph/nodes.py
from src.graph.state import AnalysisState

async def plan_node(state: AnalysisState) -> dict:
    """Chief Agent 规划节点：解析 query，选择 Expert"""
    ...

async def execute_node(state: AnalysisState) -> dict:
    """并行执行 Expert Agent 节点"""
    ...

async def evaluate_node(state: AnalysisState) -> dict:
    """质量评估节点"""
    ...

async def refine_node(state: AnalysisState) -> dict:
    """基于评估反馈重试节点"""
    ...

async def approval_node(state: AnalysisState) -> dict:
    """人工审批检查点"""
    ...

async def report_node(state: AnalysisState) -> dict:
    """生成最终报告节点"""
    ...
```

**Step 3: Commit**

```bash
git add src/graph/state.py src/graph/nodes.py
git commit -m "feat: add LangGraph AnalysisState schema and node skeletons"
```

---

### Task 2: 构建 StateGraph DAG

**Files:**
- Modify: `src/graph/nodes.py`（实现各节点）
- Create: `src/graph/graph.py`（Graph 构建）
- Create: `src/graph/__init__.py`

**Step 1: 实现 plan_node**

plan_node 调用 LLM 解析用户 query，确定需要哪些 Expert，生成执行计划。

```python
# src/graph/nodes.py (plan_node 部分)
from langchain_openai import ChatOpenAI
from src.core.config import settings
from src.graph.state import AnalysisState, AnalysisStatus
import json

async def plan_node(state: AnalysisState) -> dict:
    llm = ChatOpenAI(model=settings.llm_model, temperature=0)

    expert_mapping = {
        "用户获取": "prospect",
        "转化优化": "conversion",
        "补贴策略": "subsidy",
        "用户留存": "retention",
        "广告投放": "ad",
    }

    prompt = f"""分析以下用户需求，确定需要哪些专家参与分析。

用户需求: {state['query']}
分析范围: {state.get('scope', '综合分析')}
预算: {state.get('budget', '未指定')}

可选专家: {list(expert_mapping.keys())}

返回 JSON:
{{"selected_experts": ["专家名", ...], "plan": "执行计划描述"}}"""

    response = await llm.ainvoke(prompt)
    result = json.loads(response.content)

    return {
        "selected_experts": result["selected_experts"],
        "plan": result.get("plan", ""),
        "status": AnalysisStatus.EXECUTING,
    }
```

**Step 2: 实现 execute_node（并行 Expert 调用）**

```python
import asyncio
from src.core.expert import ExpertAgent

_expert_registry = {
    "prospect": "prospect_expert",
    "conversion": "conversion_expert",
    "subsidy": "subsidy_expert",
    "retention": "retention_expert",
    "ad": "ad_expert",
}

async def execute_node(state: AnalysisState) -> dict:
    selected = state.get("selected_experts", [])
    tasks = []
    for expert_name in selected:
        if expert_name in _expert_registry:
            agent = ExpertAgent(_expert_registry[expert_name])
            tasks.append(agent.analyze(state["query"], state.get("scope"), state.get("budget")))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    expert_results = []
    errors = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append(f"{selected[i]}: {str(result)}")
        else:
            expert_results.append({"expert": selected[i], "result": result})

    return {
        "expert_results": expert_results,
        "execution_errors": errors,
    }
```

**Step 3: 构建 StateGraph**

```python
# src/graph/graph.py
from langgraph.graph import StateGraph, END
from src.graph.state import AnalysisState, AnalysisStatus
from src.graph.nodes import plan_node, execute_node, evaluate_node, approval_node, report_node

def should_refine(state: AnalysisState) -> str:
    if state.get("needs_refinement") and state.get("refinement_round", 0) < 2:
        return "execute"
    if state.get("approval_required") and not state.get("approved"):
        return "approval"
    return "report"

def should_continue_after_approval(state: AnalysisState) -> str:
    if state.get("approved") is False:
        return END
    return "report"

def build_analysis_graph() -> StateGraph:
    graph = StateGraph(AnalysisState)

    # 添加节点
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("approval", approval_node)
    graph.add_node("report", report_node)

    # 设置入口
    graph.set_entry_point("plan")

    # 添加边
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "evaluate")

    # 条件边：评估后判断
    graph.add_conditional_edges("evaluate", should_refine, {
        "execute": "execute",
        "approval": "approval",
        "report": "report",
    })

    # 条件边：审批后判断
    graph.add_conditional_edges("approval", should_continue_after_approval, {
        END: END,
        "report": "report",
    })

    graph.add_edge("report", END)

    return graph
```

**Step 4: Commit**

```bash
git add src/graph/
git commit -m "feat: implement LangGraph StateGraph DAG with conditional edges"
```

---

### Task 3: Checkpoint 和 MemorySaver 集成

**Files:**
- Modify: `src/graph/graph.py`
- Create: `src/graph/checkpoint.py`

**Step 1: 实现 PostgreSQL Checkpoint**

```python
# src/graph/checkpoint.py
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.core.config import settings

async def get_checkpointer():
    checkpointer = AsyncPostgresSaver.from_conn_string(settings.db_url)
    await checkpointer.setup()
    return checkpointer
```

**Step 2: 集成到 Graph**

```python
# src/graph/graph.py (修改 build_analysis_graph)
from src.graph.checkpoint import get_checkpointer

async def build_compiled_graph():
    graph = build_analysis_graph()
    checkpointer = await get_checkpointer()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["approval"],  # Human-in-the-Loop: 审批前暂停
    )
```

**Step 3: Commit**

```bash
git add src/graph/checkpoint.py src/graph/graph.py
git commit -m "feat: add PostgreSQL checkpoint with human-in-the-loop approval"
```

---

## Phase 2: 安全护栏体系（Tasks 4-5）

### Task 4: 输入/输出护栏

**Files:**
- Create: `src/guardrails/__init__.py`
- Create: `src/guardrails/input_guard.py`
- Create: `src/guardrails/output_guard.py`

**Step 1: 输入护栏 - Prompt 注入检测 + 输入分类**

```python
# src/guardrails/input_guard.py
import re
from dataclasses import dataclass

@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    sanitized_input: str = ""

INJECTION_PATTERNS = [
    r"(?i)(ignore\s+previous|forget\s+your|you\s+are\s+now|system\s*prompt)",
    r"(?i)(pretend|act\s+as|roleplay|jailbreak)",
    r"(?i)(\<system\>|\<\/system\>|```system)",
    r"(?i)(translate.*above|summarize.*above|repeat.*above)",
]

def check_prompt_injection(text: str) -> GuardrailResult:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return GuardrailResult(
                passed=False,
                reason=f"Potential prompt injection detected: pattern '{pattern}'"
            )
    return GuardrailResult(passed=True, sanitized_input=text)

def validate_input(query: str, budget: float | None = None) -> GuardrailResult:
    if not query or len(query.strip()) < 5:
        return GuardrailResult(passed=False, reason="Query too short")

    if len(query) > 5000:
        return GuardrailResult(passed=False, reason="Query too long (max 5000 chars)")

    injection_check = check_prompt_injection(query)
    if not injection_check.passed:
        return injection_check

    if budget is not None and (budget < 0 or budget > 1_000_000):
        return GuardrailResult(passed=False, reason="Budget out of valid range")

    return GuardrailResult(passed=True, sanitized_input=query.strip())
```

**Step 2: 输出护栏 - 事实性校验 + 格式验证**

```python
# src/guardrails/output_guard.py
import json
import re
from dataclasses import dataclass

@dataclass
class OutputGuardResult:
    passed: bool
    reason: str = ""
    sanitized_output: str = ""

SENSITIVE_PATTERNS = [
    r"(?i)(password|api[_-]?key|secret|token)\s*[:=]\s*\S+",
    r"\b\d{16}\b",  # 信用卡号
    r"(?i)(ssn|social\s+security)\s*[:=]?\s*\d{3}-\d{2}-\d{4}",
]

def check_sensitive_info(text: str) -> OutputGuardResult:
    for pattern in SENSITIVE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return OutputGuardResult(
                passed=False,
                reason=f"Sensitive information detected in output"
            )
    return OutputGuardResult(passed=True)

def validate_output(output: str) -> OutputGuardResult:
    if not output or len(output.strip()) < 10:
        return OutputGuardResult(passed=False, reason="Output too short")

    sensitive_check = check_sensitive_info(output)
    if not sensitive_check.passed:
        return sensitive_check

    return OutputGuardResult(passed=True, sanitized_output=output.strip())
```

**Step 3: 将护栏集成到 Graph 节点**

修改 `src/graph/nodes.py` 的 `plan_node`，在调用 LLM 前加输入护栏；修改 `report_node`，输出前加输出护栏。

**Step 4: Commit**

```bash
git add src/guardrails/
git commit -m "feat: add input/output guardrails with prompt injection detection"
```

---

### Task 5: Circuit Breaker 和 Rate Limiting

**Files:**
- Create: `src/core/circuit_breaker.py`
- Modify: `src/web.py`（添加 Rate Limiter 中间件）

**Step 1: 实现 Circuit Breaker**

```python
# src/core/circuit_breaker.py
import asyncio
import time
from enum import Enum
from functools import wraps

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60, half_open_max=1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0
        self.half_open_count = 0

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_count = 0
                else:
                    raise Exception(f"Circuit breaker OPEN for {func.__name__}")

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_count >= self.half_open_max:
                    raise Exception(f"Circuit breaker HALF_OPEN limit for {func.__name__}")
                self.half_open_count += 1

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise

        return wrapper

    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

**Step 2: 在 FastAPI 添加 Rate Limiting**

```python
# 在 src/web.py 中添加
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# 应用到分析端点
@app.post("/analyze")
@limiter.limit("10/minute")
async def create_analysis(...):
    ...
```

**Step 3: Commit**

```bash
git add src/core/circuit_breaker.py src/web.py
git commit -m "feat: add circuit breaker and API rate limiting"
```

---

## Phase 3: Vector DB 记忆系统（Task 6）

### Task 6: ChromaDB + Embedding 记忆替换 TF-IDF

**Files:**
- Create: `src/memory/vector_store.py`
- Create: `src/memory/embedding.py`
- Modify: `src/memory/manager.py`（替换检索逻辑）

**Step 1: Embedding 服务**

```python
# src/memory/embedding.py
from langchain_openai import OpenAIEmbeddings
from src.core.config import settings

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.llm_api_key,
        )
    return _embeddings
```

**Step 2: ChromaDB Vector Store**

```python
# src/memory/vector_store.py
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.memory.embedding import get_embeddings

class VectorMemoryStore:
    def __init__(self, persist_dir: str = "./data/chroma"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embeddings = get_embeddings()

    def _get_collection(self, org_id: str):
        return self.client.get_or_create_collection(
            name=f"org_{org_id}",
            metadata={"hnsw:space": "cosine"},
        )

    async def store(self, org_id: str, query: str, summary: str, results: dict, tags: list[str]):
        collection = self._get_collection(org_id)
        doc_id = f"mem_{hash(query)}_{org_id}"

        embedding = await self.embeddings.aembed_query(f"{query} {summary}")

        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[f"{query}\n{summary}"],
            metadatas=[{
                "org_id": org_id,
                "tags": json.dumps(tags),
                "results_summary": json.dumps(results, ensure_ascii=False)[:2000],
            }],
        )

    async def search(self, org_id: str, query: str, top_k: int = 5) -> list[dict]:
        collection = self._get_collection(org_id)
        query_embedding = await self.embeddings.aembed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        memories = []
        for i, doc in enumerate(results["documents"][0]):
            memories.append({
                "content": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return memories
```

**Step 3: 修改 MemoryManager 使用 Vector Store**

修改 `src/memory/manager.py`，将 TF-IDF 检索替换为 `VectorMemoryStore.search()`，保留 org_id 强制过滤。

**Step 4: Commit**

```bash
git add src/memory/
git commit -m "feat: replace TF-IDF memory with ChromaDB vector store"
```

---

## Phase 4: API 和集成升级（Tasks 7-8）

### Task 7: Human-in-the-Loop API 端点

**Files:**
- Modify: `src/web.py`

**Step 1: 添加审批和恢复端点**

```python
# 在 src/web.py 中添加

@app.get("/analyses/{analysis_id}/state")
async def get_analysis_state(analysis_id: str, user=Depends(get_current_user)):
    """获取当前分析状态（含 Checkpoint）"""
    config = {"configurable": {"thread_id": analysis_id}}
    state = await graph.aget_state(config)
    return {"status": state.values.get("status"), "plan": state.values.get("plan")}

@app.post("/analyses/{analysis_id}/approve")
async def approve_analysis(analysis_id: str, approved: bool, user=Depends(get_current_user)):
    """人工审批：通过或拒绝"""
    config = {"configurable": {"thread_id": analysis_id}}
    state = await graph.aget_state(config)
    await graph.aupdate_state(config, {"approved": approved, "status": AnalysisStatus.REPORTING if approved else AnalysisStatus.FAILED})
    result = await graph.ainvoke(None, config)
    return {"status": "approved" if approved else "rejected"}

@app.post("/analyses/{analysis_id}/resume")
async def resume_analysis(analysis_id: str, user=Depends(get_current_user)):
    """从 Checkpoint 恢复中断的分析"""
    config = {"configurable": {"thread_id": analysis_id}}
    result = await graph.ainvoke(None, config)
    return {"status": "resumed", "report": result.get("final_report")}
```

**Step 2: Commit**

```bash
git add src/web.py
git commit -m "feat: add human-in-the-loop approval and resume endpoints"
```

---

### Task 8: 结构化 SSE 状态推送

**Files:**
- Modify: `src/web.py`（流式端点）

**Step 1: 定义结构化事件类型**

```python
# 在 src/api_models.py 中添加
from pydantic import BaseModel
from typing import Optional

class AgentEvent(BaseModel):
    type: str  # "plan", "execute", "evaluate", "approval", "report", "error"
    expert: Optional[str] = None
    data: dict
    progress: float  # 0.0 to 1.0
```

**Step 2: 修改 SSE 端点推送结构化事件**

将当前的纯文本 SSE 改为发送 `AgentEvent` JSON 事件，前端可以据此渲染进度条和各阶段状态。

**Step 3: Commit**

```bash
git add src/web.py src/api_models.py
git commit -m "feat: structured SSE events with agent stage tracking"
```

---

## Phase 5: 测试和清理（Tasks 9-10）

### Task 9: Graph 集成测试

**Files:**
- Create: `tests/test_graph/test_state.py`
- Create: `tests/test_graph/test_graph_execution.py`
- Create: `tests/test_guardrails/test_input.py`
- Create: `tests/test_guardrails/test_output.py`

**测试覆盖：**
- State 序列化/反序列化
- Graph 完整执行流程
- 条件边路由（正常/需重试/需审批）
- Checkpoint 保存和恢复
- 输入护栏（注入检测、长度校验、预算范围）
- 输出护栏（敏感信息、格式验证）
- Circuit Breaker（故障阈值、恢复、半开状态）

**Step: Commit**

```bash
git add tests/
git commit -m "test: add graph, guardrails, and circuit breaker tests"
```

---

### Task 10: 废弃旧代码 + 依赖更新

**Files:**
- Modify: `pyproject.toml`
- Clean up deprecated code paths

**Step 1: 更新依赖**

在 `pyproject.toml` 中添加：
- `langgraph >= 0.2`
- `langgraph-checkpoint-postgres >= 0.1`
- `chromadb >= 0.5`
- `slowapi >= 0.1`

**Step 2: 标记旧代码路径为 deprecated**

将 `src/core/chief.py` 中的旧 ReAct 循环标记为 `_deprecated`，保留向后兼容但引导新调用走 Graph 路径。

**Step 3: Commit**

```bash
git add pyproject.toml src/core/chief.py
git commit -m "chore: update deps, deprecate old ReAct loop in favor of LangGraph"
```

---

## 实施优先级

| 优先级 | Phase | 预期收益 |
|--------|-------|---------|
| P0 | Phase 1 (Tasks 1-3) | 核心架构升级，DAG + Checkpoint + HITL |
| P0 | Phase 2 Task 4 | 安全护栏，防止 Prompt 注入 |
| P1 | Phase 3 (Task 6) | 记忆系统升级，语义检索 |
| P1 | Phase 2 Task 5 | Circuit Breaker + Rate Limiting |
| P2 | Phase 4 (Tasks 7-8) | API 用户体验升级 |
| P2 | Phase 5 (Tasks 9-10) | 测试覆盖和清理 |

## 执行选择

1. **Subagent-Driven（本会话）** - 逐 Task 派发子 Agent，每 Task 后审查
2. **Parallel Session（新会话）** - 在新会话中使用 executing-plans 批量执行
