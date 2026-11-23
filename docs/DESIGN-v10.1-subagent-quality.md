# DESIGN-v10.1: 子 Agent 质量标准（Industry-Grade Sub-Agent Quality）

> 目标：每个领域 Agent 达到业界顶级水准——不是"能跑"，是"能在面试中被逐行审视"。
> 依据：Anthropic《Building Effective Agents》(2024.12) + 《Multi-Agent Research System》(2025.06)
> + LangChain《Choosing the Right Multi-Agent Architecture》(2026.01)

## 1. 业界顶级 Sub-Agent 的五项能力

综合 Anthropic 7 种设计模式 + LangChain 4 种架构模式，一个顶级的子 Agent 应具备：

| # | 能力 | 来源 | 当前状态 | 目标 |
|---|---|---|---|---|
| 1 | **ACI 工具规格**（Agent-Computer Interface） | Anthropic: "Tool design and selection are critical" | 部分（tools.py 有规格但不够完整） | 每个工具 4 要素：purpose / when-to-call / args / **failure-modes** |
| 2 | **推理循环**（ReAct with bounded steps） | Anthropic: "Agents are stateful and errors compound" | 仅 SubsidyDeepAgent PoC | 所有 DeepAgent 都有 ≤10 步 ReAct + 幂等缓存 |
| 3 | **自检 + 降级**（Self-evaluation + graceful degradation） | Anthropic: "Let agents improve themselves" | KPI 闸门在全局层 | Agent 级自检：返回前校验 KPI 合理性，异常自动降级 |
| 4 | **结构化上下文管理** | Anthropic: "Context engineering" | params 直接传递 | 结构化 DomainContext（领域知识 + 历史结果 + 兄弟 KPI） |
| 5 | **步级观测 + token 归因** | Anthropic: "Debugging benefits from new approaches" | Chief 有步级观测 | DeepAgent 每步也有观测（thought/action/obs/gate/cost） |

## 2. 升级方案

### 2.1 ACI 工具规格（能力 #1）

每个工具的规格从"名字+描述"升级为 Anthropic 推荐的完整 ACI：

```python
@dataclass
class ToolSpec:
    """Agent-Computer Interface——LLM 看到的工具完整画像。"""
    name: str                    # 工具标识
    purpose: str                 # 做什么（一句话）
    when_to_call: str            # 什么时候该调（vs 其他工具）
    args: dict[str, str]         # 参数说明（key → 描述）
    failure_modes: list[str]     # 已知失败模式（LLM 需要知道才能规避）
    output_schema: dict          # 输出结构（Pydantic model）
    cost_hint: str               # 成本提示（"轻量" / "需要训练模型，较慢"）
```

**为什么重要**（Anthropic 原文）：
> "Agent-tool interfaces are as critical as human-computer interfaces. Bad tool descriptions can send agents down completely wrong paths."
> "We gave our agents explicit heuristics: examine all available tools first, match tool usage to user intent."

### 2.2 推理循环（能力 #2）

现有 DeepAgentBase 的 ReAct 循环已实现骨架，需补充：

```python
class DeepAgentBase(BaseAgent):
    # 已有: max_deep_steps, _deep_steps, duplicate 检测
    # 新增:
    - _idempotent_cache: dict[str, Any]     # 相同参数的管线调用缓存
    - _refinement_budget: int               # 迭代预算（与步数分开计）
    - _quality_threshold: float             # 结果质量阈值（低于则重试）
```

### 2.3 Agent 级自检（能力 #3）

在 `_compile_final` 返回前加自检环节：

```python
async def _self_check(self, result: dict) -> dict:
    """Agent 级出口自检——校验 KPI 合理性，异常标记降级。"""
    kpis = result.get("kpis", {})
    issues = []
    for key, value in kpis.items():
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                issues.append(f"{key}=NaN/Inf")
            if abs(value) > 1e12:
                issues.append(f"{key}={value:.2e} (extreme)")
    if issues:
        result["self_check"] = {"status": "warn", "issues": issues}
        result["degradations"] = result.get("degradations", []) + ["deep:self-check-warn"]
    else:
        result["self_check"] = {"status": "pass"}
    return result
```

### 2.4 结构化上下文（能力 #4）

```python
@dataclass
class DomainContext:
    """Agent 收到的结构化上下文——不是裸参数。"""
    task: str                       # 任务描述
    budget: float | None            # 预算约束
    sibling_kpis: dict[str, Any]    # 兄弟专家的 KPI（跨域协作）
    prior_findings: list[dict]      # 本会话中该领域的先前发现
    domain_constraints: list[str]   # 领域特有约束（如"补贴不能为负"）
```

### 2.5 步级观测（能力 #5）

每步增加 token 成本归因：

```python
{
    "step": 1,
    "thought": "先跑标准管线",
    "action": "run_pipeline",
    "obs": "ATE=0.083, allocated=42000",
    "gate": "new",
    "ms": 1250,
    "tokens_in": 850,             # 新增
    "tokens_out": 120,            # 新增
    "cost_usd": 0.0021,           # 新增
}
```

## 3. 实施计划

| 步骤 | 内容 | 文件 | 预估 |
|---|---|---|---|
| 1 | ToolSpec dataclass + 工具规格注册 | `src/core/tool_spec.py` | 新文件 |
| 2 | DeepAgentBase 升级（自检 + 上下文 + 观测） | `src/agents/deep/base.py` | 修改 |
| 3 | SubsidyDeepAgent 适配新接口 | `src/agents/deep/subsidy.py` | 修改 |
| 4 | ProspectDeepAgent（第 2 个 PoC） | `src/agents/deep/prospect.py` | 新文件 |
| 5 | RetentionDeepAgent（第 3 个 PoC） | `src/agents/deep/retention.py` | 新文件 |
| 6 | 测试 | `tests/test_deep_agent_v2.py` | 新文件 |

## 4. 判断"业界顶级"的验收标准

一个子 Agent 达到业界顶级水准，当且仅当：

1. **工具规格完整**：每个工具的 ACI 四要素齐全，LLM 只看规格就知道怎么用
2. **推理循环有界**：步数硬上限 + 幂等缓存 + 整批重复收手，不存在无限循环可能
3. **结果可信**：出口有自检（KPI 合理性 + NaN/Inf 检查），全局有 KPI 闸门
4. **上下文结构化**：Agent 收到的不是裸 dict，是带领域约束的 DomainContext
5. **全程可观测**：每步四元组 + 耗时 + token 成本，`--json` 可完整回放
6. **零 LLM 可降级**：LLM 不可用时退化为确定性管线，功能不丢（只是少了叙述）
7. **测试覆盖**：每个 Agent 至少 5 个测试用例（零 LLM / 步数上限 / 自检 / 降级 / 观测）
