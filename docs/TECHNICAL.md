# GrowthPilot Agent — 技术架构文档

> 版本：v2.0 | 日期：2026-04-21
> 状态：优化升级版

---

## 一、系统架构

### 1.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户接入层                             │
│   CLI (Click + Rich)     │     Web API (FastAPI + SSE)      │
└─────────────┬───────────────────────┬───────────────────────┘
              │                       │
┌─────────────┴───────────────────────┴───────────────────────┐
│                      编排引擎层                               │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Orchestrator Agent                         │  │
│  │  · LLM 意图理解 (power 模型)                            │  │
│  │  · 任务分解 + Agent 调度                                 │  │
│  │  · asyncio.gather 真并行执行                            │  │
│  │  · 结果聚合 + KPI 快照                                  │  │
│  │  · 事件流进度推送                                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Hook 管理器 (新增)                          │  │
│  │  · PreRunHook  → 日志、tracing、state 增强              │  │
│  │  · PostRunHook → 指标收集、自评、result 增强            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              事件总线 (新增)                             │  │
│  │  · AgentStarted  · AgentProgress  · AgentCompleted     │  │
│  │  · ToolCalled    · ToolCompleted   · ErrorOccurred     │  │
│  └────────────────────────────────────────────────────────┘  │
├──────────┬──────────┬──────────┬──────────┬──────────────────┤
│Prospect  │Conversion│ Subsidy  │Retention │  AdAcquisition   │
│Agent     │Agent     │Agent     │Agent     │  Agent           │
│(fast/    │(default  │(default  │(default  │  (default 模型)  │
│default)  │ 模型)    │ 模型)    │ 模型)    │                  │
├──────────┴──────────┴──────────┴──────────┴──────────────────┤
│                      工具层                                   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              ToolRegistry (新增)                       │    │
│  │  · @register 装饰器注册                                │    │
│  │  · 工厂模式创建                                        │    │
│  │  · SecureDataLoader 安全加载                           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  Prospect  │ Conversion │ Subsidy  │ Retention  │ Ad  │ Common│
│  Tools     │ Tools      │ Tools    │ Tools      │Tools│ Tools │
├───────────────────────────────────────────────────────────────┤
│                     中间件层 (增强)                             │
│                                                               │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌────────────────┐  │
│  │Retry    │ │Tracing   │ │Memory     │ │Prompt Template │  │
│  │Middleware│ │Middleware│ │Manager    │ │Engine          │  │
│  │(tenacity)│ │(新增)    │ │(新增)     │ │(Jinja2 激活)   │  │
│  └─────────┘ └──────────┘ └───────────┘ └────────────────┘  │
├───────────────────────────────────────────────────────────────┤
│                     基础设施层                                  │
│                                                               │
│  Config (YAML+.env) │ LLM Factory (多模型分层) │ State       │
│  Model Tiers:       │  power: gpt-4o           │ (TypedDict) │
│  fast/default/power │  default: gpt-4o-mini    │             │
│                     │  fast: deepseek-chat     │             │
└───────────────────────────────────────────────────────────────┘
```

### 1.2 数据流

```
用户查询
  │
  ▼
Orchestrator ──→ LLM 意图理解 (power 模型)
  │
  ├──→ asyncio.gather([prospect, subsidy, ad])  ← 真并行
  │     │
  │     ├─ ProspectAgent (default 模型)
  │     │   ├─ Hook: pre_run (tracing, logging)
  │     │   ├─ Tool: FeatureEngine → IntentModel → UserScorer → Segmentor
  │     │   ├─ LLM: 结构化 Prompt → CoT 推理 → JSON 输出
  │     │   └─ Hook: post_run (metrics, self_eval)
  │     │
  │     ├─ SubsidyAgent (default 模型)
  │     │   └─ (同上流程)
  │     │
  │     └─ AdAgent (default 模型)
  │         └─ (同上流程)
  │
  ├──→ ConversionAgent (依赖并行结果)
  │
  ├──→ RetentionAgent (依赖 conversion 结果)
  │
  └──→ ReportGen
      │
      ▼
  KPI Snapshot + Strategy Recommendation
```

---

## 二、核心组件设计

### 2.1 BaseAgent (增强版)

```python
class BaseAgent(ABC):
    name: str
    description: str
    model_tier: str = "default"  # fast / default / power

    # Hook 支持
    _pre_hooks: list[PreRunHook]
    _post_hooks: list[PostRunHook]

    # 记忆支持
    _memory: GrowthMemory | None

    async def run(self, state: AgentState) -> dict[str, Any]:
        # 1. Pre hooks
        for hook in self._pre_hooks:
            state = await hook.on_pre_run(self.name, state)

        # 2. 核心执行
        result = await self._execute(state)

        # 3. Post hooks
        for hook in self._post_hooks:
            result = await hook.on_post_run(self.name, result, state)

        return result

    @abstractmethod
    async def _execute(self, state: AgentState) -> dict[str, Any]:
        ...

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _invoke_llm(self, message: str, tier: str | None = None) -> str:
        llm = create_llm(tier=tier or self.model_tier)
        ...
```

### 2.2 Hook 系统

```python
# src/core/hooks.py

class PreRunHook(ABC):
    @abstractmethod
    async def on_pre_run(self, agent_name: str, state: dict) -> dict: ...

class PostRunHook(ABC):
    @abstractmethod
    async def on_post_run(self, agent_name: str, result: dict, state: dict) -> dict: ...


# 内置 Hook 实现

class TracingHook(PostRunHook):
    """记录 Agent 执行耗时和状态变化"""
    async def on_post_run(self, agent_name, result, state):
        elapsed = time.time() - state.get("_start_time", time.time())
        result["_trace"] = {"agent": agent_name, "elapsed_s": round(elapsed, 2)}
        return result

class LoggingHook(PreRunHook, PostRunHook):
    """统一的日志输出"""
    async def on_pre_run(self, agent_name, state):
        logger.info(f"[{agent_name}] Starting execution")
        return state
    async def on_post_run(self, agent_name, result, state):
        logger.info(f"[{agent_name}] Completed. Keys: {list(result.keys())}")
        return result

class MetricsHook(PostRunHook):
    """收集执行指标"""
    async def on_post_run(self, agent_name, result, state):
        metrics = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "has_errors": bool(result.get("errors")),
        }
        result["_metrics"] = metrics
        return result
```

### 2.3 模型分层

```python
# src/core/config.py 新增

class ModelTier(BaseModel):
    provider: str
    model: str
    temperature: float = 0.5
    max_tokens: int = 4096

class Settings(BaseModel):
    # 现有字段...

    # 新增：模型分层
    model_tiers: dict[str, ModelTier] = {
        "fast": ModelTier(provider="deepseek", model="deepseek-chat", temperature=0.3),
        "default": ModelTier(provider="openai", model="gpt-4o-mini", temperature=0.5),
        "power": ModelTier(provider="openai", model="gpt-4o", temperature=0.7),
    }

# src/core/llm_factory.py 修改

def create_llm(*, tier: str = "default", settings=None) -> BaseChatModel:
    s = settings or get_settings()
    tier_config = s.model_tiers.get(tier, s.model_tiers["default"])
    return _create_from_tier(tier_config)
```

**Agent 模型分配**：

| Agent | 模型层级 | 原因 |
|-------|---------|------|
| Orchestrator | power | 需要综合推理和多维度分析 |
| ProspectAgent | default | 标准分析任务 |
| ConversionAgent | default | 标准分析任务 |
| SubsidyAgent | default | 标准分析任务 |
| RetentionAgent | default | 标准分析任务 |
| AdAgent | default | 标准分析任务 |
| JSON 解析/格式化 | fast | 简单格式化任务 |
| 自评打分 | fast | 简单评分任务 |

### 2.4 事件系统

```python
# src/core/events.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class EventStatus(str, Enum):
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentEvent:
    agent: str
    status: EventStatus
    step: str = ""
    progress: int = 0  # 0-100
    detail: str = ""
    timestamp: float = field(default_factory=time.time)

# State 扩展
class AgentState(TypedDict):
    ...
    events: Annotated[list[dict], operator.add]  # 新增

# 在 workflow 节点中使用
async def prospect_node(state: AgentState) -> dict[str, Any]:
    events = [{"agent": "prospect", "status": "started"}]
    try:
        features = engine.build_feature_matrix(data)
        events.append({"agent": "prospect", "step": "features", "progress": 30})

        scores = model.predict(features)
        events.append({"agent": "prospect", "step": "scoring", "progress": 70})

        analysis = await agent._invoke_llm(prompt)
        events.append({"agent": "prospect", "status": "completed", "progress": 100})

        return {"prospect_results": result, "events": events}
    except Exception as exc:
        events.append({"agent": "prospect", "status": "failed", "detail": str(exc)})
        return {"errors": [f"prospect: {exc}"], "events": events}
```

### 2.5 记忆系统

```python
# src/core/memory.py

class GrowthMemory:
    """跨会话的增长策略记忆"""

    def __init__(self, base_path: str = "./data/memory"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _category_path(self, category: str) -> Path:
        p = (self.base_path / category).resolve()
        # 安全校验：防止路径遍历
        p.relative_to(self.base_path.resolve())
        return p

    async def save_insight(self, category: str, key: str, insight: str):
        """保存业务洞察到文件"""
        dir_path = self._category_path(category)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{key}.md"
        file_path.write_text(insight, encoding="utf-8")

    async def recall(self, category: str, context: str = "", top_k: int = 3) -> list[dict]:
        """召回相关记忆（简单关键词匹配，可升级为向量搜索）"""
        dir_path = self._category_path(category)
        if not dir_path.exists():
            return []
        results = []
        for f in sorted(dir_path.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
            content = f.read_text(encoding="utf-8")
            results.append({"key": f.stem, "content": content})
            if len(results) >= top_k:
                break
        return results

    async def save_strategy_result(self, strategy_id: str, metrics: dict):
        """保存策略执行效果"""
        dir_path = self._category_path("strategy_results")
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{strategy_id}.json"
        file_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

# 在 Agent 中使用记忆
class ProspectAgent(BaseAgent):
    async def _execute(self, state):
        # 召回历史洞察
        memories = await self._memory.recall("seasonal", context=query)
        seasonal_context = "\n".join(m["content"] for m in memories)

        # ... 执行分析 ...

        # 保存新洞察
        if analysis.get("seasonal_insight"):
            await self._memory.save_insight("seasonal", f"prospect_{date}", insight)

        return result
```

### 2.6 工具注册表

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

    @classmethod
    def list_tools(cls) -> list[str]:
        return list(cls._tools.keys())


# 使用示例
@ToolRegistry.register("feature_engine")
class FeatureEngine:
    ...

# 在 Agent 中使用
class ProspectAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm=llm)
        self._feature_engine = ToolRegistry.create("feature_engine")
        self._intent_model = ToolRegistry.create("intent_model")
```

### 2.7 安全数据加载器

```python
# src/tools/common/secure_loader.py

class SecureDataLoader:
    ALLOWED_EXTENSIONS = {".csv", ".parquet", ".pq", ".json"}
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    def validate_path(self, path: str) -> Path:
        p = Path(path).resolve()
        # 路径遍历防护
        if self.base_dir:
            p.relative_to(self.base_dir)
        # 文件类型校验
        if p.suffix not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"不支持的文件类型: {p.suffix}，仅支持 {self.ALLOWED_EXTENSIONS}")
        # 文件大小校验
        if p.exists() and p.stat().st_size > self.MAX_FILE_SIZE:
            raise ValueError(f"文件过大 ({p.stat().st_size / 1024 / 1024:.0f}MB)，上限 500MB")
        return p

    def load(self, path: str) -> dict:
        p = self.validate_path(path)
        if p.suffix == ".csv":
            df = pd.read_csv(p)
        elif p.suffix in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        elif p.suffix == ".json":
            df = pd.read_json(p)
        return {"user_logs": df, "user_profile": pd.DataFrame()}
```

---

## 三、项目结构（优化后）

```
growth-pilot-agent/
├── docs/
│   ├── DESIGN.md                    # 系统设计文档
│   ├── PRD.md                       # 产品需求文档
│   ├── TECHNICAL.md                 # 技术架构文档
│   └── plans/
│       └── 2026-04-21-optimization-plan.md  # 优化方案
├── src/
│   ├── agents/
│   │   ├── orchestrator.py          # 编排 Agent (增强: 真并行, 动态调度)
│   │   ├── prospect.py              # 潜客 Agent (增强: 结构化 Prompt)
│   │   ├── conversion.py            # 转化 Agent (增强: 结构化 Prompt)
│   │   ├── subsidy.py               # 补贴 Agent (增强: 结构化 Prompt)
│   │   ├── retention.py             # 留存 Agent (增强: 结构化 Prompt)
│   │   └── ad.py                    # 广告 Agent (增强: 结构化 Prompt)
│   ├── core/
│   │   ├── base.py                  # BaseAgent (增强: Hook, 重试, 模型分层)
│   │   ├── config.py                # 配置 (增强: 模型分层配置)
│   │   ├── hooks.py                 # Hook 系统 (新增)
│   │   ├── events.py                # 事件定义 (新增)
│   │   ├── memory.py                # 持久化记忆 (新增)
│   │   ├── llm_factory.py           # LLM 工厂 (增强: 模型分层)
│   │   └── state.py                 # 状态定义 (增强: events 字段)
│   ├── graph/
│   │   └── workflow.py              # LangGraph 工作流 (增强: 事件推送)
│   ├── middleware/
│   │   └── retry.py                 # 重试中间件 (新增/重构)
│   ├── prompts/
│   │   ├── config/                  # Prompt 配置
│   │   └── templates/
│   │       ├── agents/              # 各 Agent 的 Prompt 模板 (激活)
│   │       └── components/          # 通用 Prompt 组件 (激活)
│   ├── tools/
│   │   ├── registry.py              # 工具注册表 (新增)
│   │   ├── common/
│   │   │   ├── data_loader.py       # 数据加载 (增强: 安全校验)
│   │   │   ├── secure_loader.py     # 安全加载器 (新增)
│   │   │   ├── experiment_platform.py
│   │   │   └── visualizer.py
│   │   ├── prospect/
│   │   ├── conversion/
│   │   ├── subsidy/
│   │   ├── retention/
│   │   └── ad/
│   ├── report/
│   │   └── generator.py
│   ├── cli.py
│   └── web.py
├── data/
│   └── memory/                      # 持久化记忆存储 (新增)
├── tests/
├── pyproject.toml
└── Makefile
```

---

## 四、依赖变更

### 4.1 新增依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| tenacity | >=8.2 | LLM 调用重试机制 |

### 4.2 已有依赖（无需变更）

- langgraph, langchain-core — Agent 编排
- openai, langchain-openai — LLM 调用
- lightgbm, scikit-learn — ML 模型
- dowhy — 因果推断
- pulp — 整数规划
- pandas, numpy — 数据处理
- click, rich — CLI
- fastapi, sse-starlette — Web API
- jinja2 — Prompt 模板

---

## 五、接口变更

### 5.1 AgentState 新增字段

```python
class AgentState(TypedDict):
    # 现有字段 (不变)
    query: str
    data_path: str
    budget: NotRequired[float]
    scope: NotRequired[str]
    prospect_results: NotRequired[dict[str, Any]]
    conversion_results: NotRequired[dict[str, Any]]
    subsidy_results: NotRequired[dict[str, Any]]
    retention_results: NotRequired[dict[str, Any]]
    ad_results: NotRequired[dict[str, Any]]
    experiment_results: NotRequired[dict[str, Any]]
    kpi_snapshot: NotRequired[dict[str, Any]]
    seasonal_context: NotRequired[dict[str, Any]]
    analysis_summary: NotRequired[str]
    strategy_recommendation: NotRequired[str]
    report: NotRequired[str]
    errors: Annotated[list[str], operator.add]
    metadata: Annotated[list[dict[str, Any]], operator.add]

    # 新增字段
    events: Annotated[list[dict[str, Any]], operator.add]  # 执行事件流
    memory_context: NotRequired[dict[str, Any]]              # 历史记忆上下文
```

### 5.2 BaseAgent 接口变更

```python
class BaseAgent(ABC):
    # 新增
    model_tier: str = "default"
    _pre_hooks: list[PreRunHook]
    _post_hooks: list[PostRunHook]
    _memory: GrowthMemory | None

    # 变更: run 方法增加 hook 和事件支持
    async def run(self, state: AgentState) -> dict[str, Any]: ...

    # 新增: 子类实现
    async def _execute(self, state: AgentState) -> dict[str, Any]: ...

    # 变更: _invoke_llm 支持模型分层和重试
    @retry(...)
    async def _invoke_llm(self, message: str, tier: str | None = None) -> str: ...

    # 变更: _build_prompt_context 支持记忆注入
    def _build_prompt_context(self, state: AgentState) -> str: ...
```

### 5.3 LLM Factory 接口变更

```python
# 新增 tier 参数
def create_llm(*, tier: str = "default", settings=None) -> BaseChatModel: ...
```

---

## 六、配置变更

### 6.1 .env.example 新增

```env
# 模型分层配置
FAST_PROVIDER=deepseek
FAST_MODEL=deepseek-chat
FAST_TEMPERATURE=0.3

DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0.5

POWER_PROVIDER=openai
POWER_MODEL=gpt-4o
POWER_TEMPERATURE=0.7

# 记忆系统
MEMORY_BASE_PATH=./data/memory

# 重试配置
MAX_RETRIES=3
RETRY_MIN_WAIT=1
RETRY_MAX_WAIT=10
```
