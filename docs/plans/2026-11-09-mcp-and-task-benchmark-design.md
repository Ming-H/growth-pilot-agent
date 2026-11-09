# 设计：六专家工具 MCP 化 + 任务级评测基准（v9.1）

日期：2026-11-09 ｜ 状态：定案（用户批准，方案一 A+B）

## 背景与动机

2026-11 业界调研结论：

1. **MCP 已成事实标准**（月 SDK 下载 9700 万、5800+ servers、NSA 安全指南）——工具层私有协议是孤岛，MCP 化让六专家的增长能力可被任何 MCP 宿主（Claude Code/Desktop、其他 Agent）直接调用。
2. **评测驱动是 Agent 工程师核心能力维度**（τ-bench 任务式评测、rubric judge、裁判校准）——现有三模式评测（router/heuristic/full）回答的是"管线通没通"，缺"**任务答对了没**"的端到端打分。

两项共同放大本项目独有哲学：**上秤（可判定）+ 对账（可追溯）**。

## B：MCP 服务（先做，1 天）

### 范围

新增 `src/mcp_server.py`：基于官方 SDK 的 FastMCP（`mcp>=1.2`），stdio 传输，`gpa mcp` 子命令启动。

### 暴露 5 个工具（全部确定性、零 LLM、合成数据 seed=42、`data_source: synthetic` 显式标注）

| 工具 | 底层 | 高层入参（Claude 可自然调用） | 返回 |
|---|---|---|---|
| `subsidy_optimize` | SubsidyAllocator/BudgetOptimizer（IP） | total_budget_yuan, min/max_coupon | 分段分配方案+预期增量订单+ROI |
| `ltv_rank` | prospect LTV 预测 | top_k | 客群 LTV 排序 |
| `churn_watchlist` | retention 流失预警 | top_k | 高危客群名单+分数 |
| `funnel_bottleneck` | conversion 漏斗 | — | 各级转化率+最大流失环节 |
| `budget_allocate` | budget 全局分配器（凹响应曲线+IP） | total_budget_yuan | 跨渠道分配+KPI 校准说明 |

设计要点：

- **高层参数**：工具吃"预算多少"级别入参，不吃 dataframe——数据由 DataLoader 合成（与 experts 的 demo 路径同一 seed，结果与图内运行一致）。
- **ACI 风格 docstring**：purpose / when-to-call / args / failure-modes，与 `src/agent/tools.py` 规格纪律一致。
- **不进图、不开 LLM、不写记忆**——MCP 层是工具层的纯暴露，闸门语义由宿主侧（或继续走 `gpa analyze`）承担。诚实边界写进每个工具描述。
- 依赖入主依赖组（`mcp` 轻，无重传递）。

### 测试（TDD）

- 单测：5 个工具函数在 demo 数据上的返回结构/数值稳定性（复用现有确定性纪律）。
- 协议级集成：`mcp` SDK 的 in-memory `Client` ↔ FastMCP `Server`——`list_tools()` 校验注册、`call_tool()` 校验真实往返，无网络、无 stdio、CI 安全。

## A：任务级评测基准（τ-bench for Growth）（2-3 天）

### 范围

新增 `src/evals/tasks.py`（任务定义+真值）与 `gpa eval --tasks` 运行器。

### 任务形态（首批 7 个）

每任务 = `{id, query, expected_experts, ground_truth, tolerance, kind}`：

1. `subsidy_plan` —"预算 X 的补贴怎么分" → 真值=BudgetOptimizer 直调结果（测管线把最优数送达报告）
2. `ltv_top` —"哪类用户 LTV 最高" → 真值=直调 LTV 工具的排序第一名
3. `churn_top` —流失最高危客群 → 直调真值
4. `funnel_worst` —漏斗最大流失环节 → 直调真值
5. `budget_channels` —跨渠道分配 → 直调真值
6. `multi_domain` —复合问题（须路由 ≥2 专家）
7. `honesty_refusal` —超出六领域的不可答问题 → 期望**诚实拒答/降级**（不许编数字）——把"宁拒不编"写成可评测约束

真值全部由**绕过图的直调工具**确定性生成（seed 同源）——测的是"路由→执行→报告→闸门"全链路送达率，不是优化器本身。

### 评分（三项全过才成功）

1. **路由正确**：expected_experts ⊆ 实际执行专家
2. **数值命中**：真值出现在 kpi_snapshot ∪ 报告（复用 `kpi_gate._values_match` 容差匹配——评测与闸门共用同一数值匹配语义，避免两把尺子）
3. **闸门通过**：kpi_gate_result 无 degraded（或诚实任务：报告不含 KPI 断言）

输出：逐任务表（✓/✗ + 失败原因）+ 总成功率；`--min-tasks` 门槛跌破非零退出（对齐 --router CI 纪律）；默认进 CI。

### 纪律

- 全程零 LLM（heuristic 路径），字节可复现，CI 安全；LLM 模式的任务评测留给 `--full` 扩展（rubric judge，另行设计裁判校准）。
- 评测不改图、不改闸门——只读运行结果。

## 明确不做（YAGNI）

- 不做 MCP 资源/提示原语（tools 足够）
- 不做 SSE/HTTP 传输的 MCP server（stdio 先行，宿主覆盖面已够）
- 不动 agent 模式图（子代理隔离重构另行立项）
- 不做 OTel 导出（本批不做，留 v9.2 候选）

## 验收

- `uv run pytest`：574+ 新增全绿
- `gpa mcp` 起服务，`claude mcp add` 后 Claude Code 可列工具并调用 `subsidy_optimize`
- `gpa eval --tasks` 输出 7 任务成功表，成功率 100%（诚实任务按设计）
- CI 新增 tasks 评测步骤绿灯；README/CLAUDE.md/CHANGELOG 同步（v9.1.0）
