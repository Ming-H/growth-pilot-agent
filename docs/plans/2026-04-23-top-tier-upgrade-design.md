# GrowthPilot Agent — 顶级 Agent 项目升级方案

> 日期：2026-04-23
> 版本：v5.0
> 对标：OpenAI Agents SDK / Anthropic Agent Patterns / Google ADK / LangGraph / CrewAI
> 目标：打造生产级 Multi-Agent 系统，覆盖 90%+ 行业最佳实践

---

## 一、升级总览

### 1.1 升级维度

| 层级 | 升级项 | 对标框架 | 优先级 |
|------|--------|---------|--------|
| **架构** | 统一为 Chief Agent + Expert Pipeline 架构 | OpenAI Agent-as-Tool + Anthropic Orchestrator-Workers | P0 |
| **交互** | Expert 返回 Pydantic 模型 + Structured Output | OpenAI Structured Output | P0 |
| **数据** | 移除硬编码 sample data，支持外部数据注入 | Production-readiness | P0 |
| **Prompt** | 独立文件 + Few-shot + 版本化 | Anthropic Prompt Engineering | P1 |
| **质量** | 实现 Evaluator-Optimizer 循环 | Anthropic 核心模式 | P1 |
| **人机** | HITL 审批门 + Checkpoint 持久化 | LangGraph + Google ADK | P1 |
| **记忆** | Embedding 向量检索替代 TF-IDF | 现代 RAG 模式 | P2 |
| **可观测** | OpenTelemetry + 按 Agent 成本归因 | LangSmith 模式 | P2 |
| **评估** | 自动化 Eval Pipeline + CI 集成 | Google ADK eval | P2 |

### 1.2 升级后的架构

```
┌──────────────────────────────────────────────────────────────┐
│                        用户层                                  │
│   CLI (Click + Rich)  │  Web API (FastAPI + SSE Streaming)   │
└─────────────┬────────────────────────────────────────────────┘
              │
┌─────────────┴────────────────────────────────────────────────┐
│                   Tier 1: Chief Agent                         │
│   ReAct Loop + Tool Calling + Guardrails + Memory Context    │
│                                                              │
│   Phase 0: Input Guardrail (validation)                      │
│   Phase 1: Plan (LLM decides which experts)                  │
│   Phase 2: Execute (parallel expert invocation)              │
│   Phase 3: Evaluate (Evaluator-Optimizer loop)               │
│   Phase 4: Report (structured synthesis)                     │
│   Phase 5: HITL Gate (approval for high-stakes decisions)    │
├──────┬──────┬──────┬──────┬──────────────────────────────────┤
│ Tier 2: Expert Agents (as Tools for Chief Agent)             │
│                                                              │
│  ProspectExpert │ ConversionExpert │ SubsidyExpert │         │
│  RetentionExpert│ AdExpert          │                          │
│                                                              │
│  每个 Expert:                                                 │
│  1. 接收 Pydantic 输入模型                                     │
│  2. 运行确定性工具管线                                         │
│  3. LLM 综合 (Structured Output)                             │
│  4. 返回 Pydantic 输出模型                                     │
├──────────────────────────────────────────────────────────────┤
│ Tier 3: Deterministic Tools (纯 Python)                       │
│                                                              │
│  ML Tools: FeatureEngine, IntentModel, CausalEngine, etc.    │
│  Common Tools: DataLoader, Visualizer, ExperimentPlatform    │
│  All tools accept data parameters (no hardcoded samples)     │
├──────────────────────────────────────────────────────────────┤
│ Cross-cutting Concerns                                        │
│                                                              │
│  Guardrails (Input/Plan/Output)  │ HITL (Approval Gates)     │
│  Checkpoint (State Persistence)  │ Memory (Vector Search)    │
│  Observability (OpenTelemetry)   │ Eval Pipeline (CI/CD)     │
│  Middleware (Retry/Logging/Trace)│ Config (Tiered Models)    │
└──────────────────────────────────────────────────────────────┘
```

---

## 二、升级项详细设计

### UPG-01: 架构统一 — 移除旧 OrchestratorAgent

**问题**: `orchestrator.py` 的 `OrchestratorAgent`（关键词路由）和 `chief.py` 的 `GrowthPilotAgent`（ReAct）功能重叠。

**方案**:
1. 删除 `src/agents/orchestrator.py` — 其职责已由 Chief Agent 完全接管
2. 清理 `AgentState` TypedDict — 移除仅旧架构使用的字段
3. `graph/workflow.py` 仅保留 `run_workflow()` 作为入口
4. 更新 `agents/__init__.py` 导出

**保留**: `detect_scope()` 的关键词映射作为 fallback，但不作为主路由机制

**影响文件**:
- `src/agents/orchestrator.py` → 删除
- `src/agents/__init__.py` → 更新导出
- `src/core/state.py` → 精简
- `src/graph/workflow.py` → 简化
- `docs/DESIGN.md` → 更新架构图

---

### UPG-02: 结构化 Agent 通信 — Pydantic 模型替代 JSON 字符串

**问题**: `ExpertAgentBase.analyze()` 返回 `json.dumps()` 字符串，Chief Agent 再 `json.loads()`。脆弱且丢失类型信息。

**方案**:

```python
# 修改前: ExpertAgentBase.analyze() returns str
async def analyze(self, params: dict | str) -> str:
    return json.dumps(results, ensure_ascii=False, default=str)

# 修改后: ExpertAgentBase.analyze() returns AgentResult 子类
async def analyze(self, params: dict | str) -> AgentResult:
    results = self._execute_pipeline(params)
    # LLM synthesis fills structured fields
    return self._build_result(results)
```

**具体变更**:
1. `ExpertAgentBase.analyze()` 返回 `AgentResult` 子类（已有 `ProspectResult`, `AdResult` 等）
2. `expert_tools.py` 的 `@tool` 函数返回 Pydantic 模型
3. Chief Agent 的 `_accumulate_context()` 直接读模型属性，不再 `json.loads()`
4. 所有 inter-agent 通信通过 Pydantic 模型

**影响文件**:
- `src/core/expert.py` — 修改 `analyze()` 签名
- `src/core/models.py` — 确保所有 Result 模型完整
- `src/tools/expert_tools.py` — 更新 tool 函数
- `src/core/chief.py` — 更新 context accumulation

---

### UPG-03: 真实数据注入 — 移除硬编码 Sample Data

**问题**: 所有 Expert Agent 内部生成 sample data（如 `AdExpert._build_sample_rta_data()`），无法对接真实数据。

**方案**:

```python
# 修改前: 硬编码
class AdExpert(ExpertAgentBase):
    def _execute_pipeline(self, params: dict) -> dict:
        historical_data = self._build_sample_rta_data()  # 硬编码
        ...

# 修改后: 从 params 或 DataLoader 获取数据
class AdExpert(ExpertAgentBase):
    def _execute_pipeline(self, params: dict) -> dict:
        data_loader = self._tools.get("data_loader", DataLoader())
        historical_data = data_loader.load(
            params.get("data_path", ""),
            fallback="generate_sample",  # 仅在无数据时生成
            seed=42,
        )
        ...
```

**原则**:
- 优先从 `params["data_path"]` 加载真实数据
- 无数据时使用 `DataLoader.generate_sample()` 作为 fallback
- 每个 tool 的 `__init__` 接受可选的 `data_loader` 参数
- Sample data 生成逻辑移到 `DataLoader` 中统一管理

**影响文件**:
- `src/agents/ad.py` — 移除 `_build_sample_*()` 方法
- `src/agents/conversion.py` — 同上
- `src/agents/prospect.py` — 同上
- `src/agents/retention.py` — 同上
- `src/agents/subsidy.py` — 同上
- `src/tools/common/data_loader.py` — 增强 `generate_sample()` 方法

---

### UPG-04: Prompt 工程升级 — 独立文件 + Few-shot

**问题**: Prompt 作为 Python 字符串常量，不可独立迭代和版本化。

**方案**:

1. **Prompt 文件结构**:
```
src/prompts/
├── chief/
│   ├── system_full.md      # Chief Agent 完整系统提示
│   ├── system_short.md     # 简化版
│   └── synthesis.md        # 综合分析提示
├── experts/
│   ├── prospect.md         # 潜客专家提示
│   ├── conversion.md       # 转化专家提示
│   ├── subsidy.md          # 补贴专家提示
│   ├── retention.md        # 留存专家提示
│   └── ad.md               # 广告专家提示
├── components/
│   ├── business_context.md # 共享业务背景
│   ├── output_format.md    # 输出格式约束
│   └── safety_rules.md     # 安全规则
└── loader.py               # Prompt 加载器
```

2. **每个 Prompt 文件包含**:
   - 角色定位（Role）
   - 业务背景（Context，支持变量注入）
   - 推理步骤（CoT guidance）
   - Few-shot 示例（1-2 个完整示例）
   - 输出格式（JSON Schema）

3. **PromptLoader**:
```python
class PromptLoader:
    """Load and render prompt templates from markdown files."""

    def __init__(self, prompts_dir: str = "src/prompts"):
        self.prompts_dir = Path(prompts_dir)

    def load(self, name: str, **variables: Any) -> str:
        """Load a prompt file and render with Jinja2 variables."""
        path = self.prompts_dir / f"{name}.md"
        template = Jinja2Template(path.read_text(encoding="utf-8"))
        # Inject common variables
        variables.setdefault("current_date", datetime.now().strftime("%Y-%m-%d"))
        variables.setdefault("current_season", self._get_season())
        return template.render(**variables)
```

4. **Few-shot 示例格式** (嵌入 prompt 文件中):
```markdown
## 示例

**输入**: "帮我分析最近的新客获取效果"

**输出**:
```json
{
  "summary": "本周新客获取 3,200 人，环比增长 12%...",
  "confidence": 0.85,
  ...
}
```
```

**影响文件**:
- 新增 `src/prompts/**/*.md` (8+ 文件)
- 新增 `src/prompts/loader.py`
- 修改 `src/core/chief.py` — 使用 PromptLoader
- 修改 `src/prompts/templates/agent_prompts.py` — 迁移到文件
- 修改所有 Expert Agent — 使用文件化 Prompt

---

### UPG-05: Evaluator-Optimizer 循环实现

**问题**: `chief.py` Phase 3 是空实现（`pass`）。

**方案**: 实现 Anthropic 的 Evaluator-Optimizer 模式。

```python
async def _evaluate_and_refine(
    self,
    expert_results: dict[str, AgentResult],
    query: str,
    *,
    max_refinement_rounds: int = 2,
    quality_threshold: float = 0.7,
) -> dict[str, AgentResult]:
    """Evaluator-Optimizer loop.

    1. Evaluate each expert result quality
    2. If quality < threshold, generate refinement feedback
    3. Re-invoke low-quality experts with feedback
    4. Repeat up to max_refinement_rounds
    """
    for round_num in range(max_refinement_rounds):
        # Step 1: Evaluate
        quality_scores = await self._batch_evaluate(expert_results, query)

        # Step 2: Check if all pass threshold
        low_quality = {
            name: score for name, score in quality_scores.items()
            if score.overall < quality_threshold
        }
        if not low_quality:
            break  # All pass

        # Step 3: Generate refinement feedback and re-invoke
        for name, score in low_quality.items():
            feedback = score.reasoning
            expert_results[name] = await self._reinvoke_expert(
                name, query, feedback=feedback
            )

    return expert_results
```

**质量评估维度**:
- 完整性 (completeness): 0-1
- 可操作性 (actionability): 0-1
- 数据支撑度 (data_grounding): 0-1
- 一致性 (consistency): 与其他专家结果是否矛盾

**影响文件**:
- `src/core/chief.py` — 实现 Phase 3
- `src/core/evaluator.py` — 新增或增强

---

### UPG-06: Human-in-the-Loop 审批门

**问题**: 高风险决策（如分配 10 万预算、全量推送）没有人工审批。

**方案**: 在 Chief Agent 的执行流程中增加审批门。

```python
class ApprovalGate:
    """Human-in-the-Loop approval gate for high-stakes decisions."""

    def __init__(self, *, auto_approve_threshold: float = 10000):
        self.auto_approve_threshold = auto_approve_threshold

    async def request_approval(
        self,
        decision_type: str,  # "budget_allocation", "campaign_launch", etc.
        details: dict[str, Any],
    ) -> ApprovalDecision:
        """Request human approval for a decision.

        Returns:
            ApprovalDecision with approved=True/False and optional modifications.
        """
        # Auto-approve low-stakes decisions
        if self._is_low_stakes(decision_type, details):
            return ApprovalDecision(approved=True, reason="Auto-approved: low stakes")

        # For high-stakes, pause and request human input
        # In CLI: prompt user directly
        # In Web: return pending state, wait for callback
        ...

@dataclass
class ApprovalDecision:
    approved: bool
    reason: str = ""
    modifications: dict[str, Any] = field(default_factory=dict)
```

**集成点**:
- Chief Agent Phase 2 → Phase 3 之间：专家结果汇总后、综合前
- 预算分配 > 阈值时
- 策略涉及全量用户时

**影响文件**:
- 新增 `src/core/approval.py`
- 修改 `src/core/chief.py` — 集成审批门
- 修改 `src/web.py` — Web 端审批回调
- 修改 `src/cli.py` — CLI 端交互式审批

---

### UPG-07: Checkpoint 状态持久化

**问题**: 执行中断后状态全部丢失。

**方案**: 轻量级 SQLite 检查点系统。

```python
class CheckpointManager:
    """Lightweight checkpoint system for agent execution state."""

    def __init__(self, db_path: str = "./data/checkpoints.db"):
        self.db_path = Path(db_path)
        self._init_db()

    async def save(self, run_id: str, state: dict[str, Any]) -> None:
        """Save current execution state."""

    async def restore(self, run_id: str) -> dict[str, Any] | None:
        """Restore a previous execution state."""

    async def list_runs(self, status: str | None = None) -> list[dict]:
        """List all saved runs."""

    async def resume(self, run_id: str) -> None:
        """Resume an interrupted run from the last checkpoint."""
```

**Checkpoint 时机**:
- 每个 Expert 调用完成后
- Evaluator 反馈后
- HITL 审批前后

**影响文件**:
- 新增 `src/core/checkpoint.py`
- 修改 `src/core/chief.py` — 集成 checkpoint
- 修改 `src/graph/workflow.py` — run_workflow 支持 resume

---

### UPG-08: 记忆系统升级 — Embedding 向量检索

**问题**: TF-IDF 字符级分词对中文效果极差。

**方案**: 使用轻量级 sentence-transformer 向量检索。

```python
class VectorMemoryManager(MemoryManager):
    """Embedding-based memory retrieval, replacing TF-IDF."""

    def __init__(
        self,
        base_path: str = "./data/memory",
        embedding_model: str = "shibing624/text2vec-base-chinese",
    ):
        # Fallback to TF-IDF if embedding model unavailable
        ...
```

**策略**:
- 优先使用 `text2vec-base-chinese` 做 embedding
- 自动 fallback 到 jieba 分词 + TF-IDF
- 记忆条目增加摘要（LLM 生成），而非存储完整结果
- 添加记忆过期和合并机制

**影响文件**:
- 修改 `src/memory/manager.py` — 升级检索
- 新增 `src/memory/embedding.py` — embedding 工具

---

### UPG-09: 可观测性 — OpenTelemetry + 成本归因

**问题**: 自定义 TraceEntry，无标准可观测性。

**方案**:

1. **OpenTelemetry Tracing**:
```python
from opentelemetry import trace

tracer = trace.get_tracer("growth-pilot-agent")

class TracingMiddleware(AgentMiddleware):
    async def wrap_model_call(self, request, handler):
        with tracer.start_as_current_span(
            f"llm.call.{request.get('agent', 'unknown')}"
        ) as span:
            span.set_attribute("model", request.get("model", ""))
            response = await handler(request)
            span.set_attribute("tokens.input", response.usage_metadata.get("input_tokens", 0))
            span.set_attribute("tokens.output", response.usage_metadata.get("output_tokens", 0))
            return response
```

2. **Per-Agent Cost Tracking**:
```python
class CostTracker:
    """Track token usage and estimated cost per agent."""

    def __init__(self):
        self._costs: dict[str, TokenUsage] = {}

    def record(self, agent_name: str, model: str, usage: dict):
        """Record token usage for an agent."""

    def report(self) -> dict[str, dict]:
        """Generate cost report per agent."""
        return {
            name: {
                "total_tokens": usage.total,
                "estimated_cost_usd": usage.total * self._get_price(model),
                "llm_calls": usage.calls,
            }
            for name, usage in self._costs.items()
        }
```

**影响文件**:
- 新增 `src/core/observability.py`
- 修改 `src/middleware/__init__.py` — 添加 OTEL middleware
- 修改 `src/core/chief.py` — 使用 CostTracker
- `pyproject.toml` — 添加 opentelemetry 依赖

---

### UPG-10: 自动化 Eval Pipeline

**问题**: 评估系统存在但无 CI 集成。

**方案**: Google ADK 风格的 eval 命令。

```bash
# 运行评估数据集
gpa eval --dataset evals/datasets/default.json

# CI 集成
gpa eval --dataset evals/datasets/default.json --min-score 0.7 --fail-below 0.5
```

**Eval 数据集格式** (`evals/datasets/default.json`):
```json
[
  {
    "id": "eval_001",
    "query": "帮我制定下周的货运增长方案",
    "scope": "full",
    "expected_agents": ["prospect", "subsidy", "ad", "conversion", "retention"],
    "expected_keys": ["analysis_summary", "strategy_recommendation", "kpi_snapshot"],
    "reference_answer": "...",
    "quality_threshold": 0.7
  }
]
```

**影响文件**:
- 修改 `src/evals/dataset.py` — 支持 JSON 文件加载
- 修改 `src/evals/evaluator.py` — batch evaluation
- 修改 `src/cli.py` — 添加 `eval` 命令
- 新增 `evals/datasets/default.json` — 默认评估数据集

---

### UPG-11: API 安全加固

**问题**: `api_key = s.llm_api_key or "sk-demo-placeholder"` 静默失败。

**方案**:
```python
# 修改前
api_key = s.llm_api_key or "sk-demo-placeholder"

# 修改后
api_key = s.llm_api_key
if not api_key:
    logger.warning("No API key configured. Running in offline/demo mode.")
    # Only allow demo mode when explicitly enabled
    if not s.demo_mode:
        raise ValueError("GPA_LLM_API_KEY is required. Set it in .env or enable GPA_DEMO_MODE=true")
```

**影响文件**:
- `src/core/config.py` — 添加 `demo_mode` 配置
- `src/core/llm_factory.py` — fail-fast 逻辑
- `src/core/guardrails.py` — input guardrail 增强

---

## 三、实施计划

### Phase 1: 核心架构 (UPG-01, UPG-02, UPG-03)

**目标**: 统一架构、结构化通信、真实数据

**任务列表**:
1. 删除 `orchestrator.py`，清理旧架构引用
2. 修改 `ExpertAgentBase.analyze()` 返回 Pydantic 模型
3. 更新 `expert_tools.py` tool 函数
4. 移除所有 Expert 的 `_build_sample_*()` 方法
5. 增强 `DataLoader` 支持 `generate_sample()` fallback
6. 更新 `chief.py` 的 context accumulation 逻辑
7. 更新所有测试

### Phase 2: 质量保障 (UPG-04, UPG-05, UPG-11)

**目标**: Prompt 升级、Evaluator-Optimizer、API 安全

**任务列表**:
1. 创建 prompt 文件目录结构
2. 迁移所有 prompt 到独立 .md 文件
3. 实现 PromptLoader
4. 实现 Evaluator-Optimizer 循环
5. 加固 API key 处理
6. 更新测试

### Phase 3: 生产就绪 (UPG-06, UPG-07, UPG-08)

**目标**: HITL、Checkpoint、记忆升级

**任务列表**:
1. 实现 ApprovalGate
2. 实现 CheckpointManager
3. 集成 HITL 到 Chief Agent
4. 升级记忆系统为 embedding 检索
5. 更新测试

### Phase 4: 可观测与评估 (UPG-09, UPG-10)

**目标**: OpenTelemetry、成本追踪、Eval Pipeline

**任务列表**:
1. 实现 OpenTelemetry middleware
2. 实现 CostTracker
3. 创建默认 eval 数据集
4. 添加 `gpa eval` CLI 命令
5. 更新文档

---

## 四、技术栈更新

```toml
# 新增依赖
dependencies = [
    # ... 现有依赖 ...
    "opentelemetry-api>=1.20",       # 可观测性
    "opentelemetry-sdk>=1.20",
    "aiosqlite>=0.19",               # Checkpoint 存储
]

[project.optional-dependencies]
embedding = [
    "sentence-transformers>=2.2",    # 向量检索
]
```

---

## 五、升级后系统特性总结

**架构**: Chief Agent (ReAct + Tool Calling) + 5 Expert Agents (Agent-as-Tool) + Pydantic 结构化通信

**质量保障**: Evaluator-Optimizer 循环 — 自动评估 Expert 输出质量，低于阈值自动重新调用并附带改进反馈

**Prompt 工程**: 文件化管理 (.md)，CoT 推理引导，few-shot 示例，动态上下文注入，版本化支持

**生产特性**: Human-in-the-Loop 审批门、Checkpoint 状态持久化（中断恢复）、OpenTelemetry 可观测性、按 Agent 成本归因

**评估体系**: 自动化 Eval Pipeline (Google ADK 风格)，LLM-as-Judge 质量评估，回归测试，CI 集成

**记忆系统**: Embedding 向量检索 (text2vec-base-chinese)，自动 fallback 到 jieba + TF-IDF
