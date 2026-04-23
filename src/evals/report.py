"""EvalReport - Generate evaluation reports in Markdown format.

Produces comprehensive evaluation reports with dimension scores,
overall assessment, and actionable improvement suggestions.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.evals.evaluator import EvalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Improvement suggestions engine
# ---------------------------------------------------------------------------

_SUGGESTIONS: dict[str, dict[str, list[str]]] = {
    "task_completion": {
        "low": [
            "检查 Agent 的工具调用链是否完整执行",
            "确认 expected_keys 与 Agent 输出格式对齐",
            "增加更多输入上下文（如用户行为数据、历史记录）",
            "优化 LLM prompt 明确要求输出所有必要字段",
        ],
        "medium": [
            "优化 prompt 使 Agent 更可靠地输出关键信息",
            "增加输出格式校验和重试机制",
            "补充缺失字段的默认值处理",
        ],
    },
    "tool_efficiency": {
        "low": [
            "审查工具调用逻辑，消除冗余调用",
            "实现工具结果缓存，避免重复计算",
            "检查工具是否正常返回，避免因错误导致的重试风暴",
        ],
        "medium": [
            "合并相似功能的工具调用",
            "优化工具调用顺序，减少依赖等待",
        ],
    },
    "output_quality": {
        "low": [
            "优化 Agent 的 system prompt，提供更清晰的任务指引",
            "增加 few-shot 示例提升输出质量",
            "使用更高能力的 LLM 模型（如从 fast 升级到 power tier）",
            "增加输出后处理步骤（格式化、去重、摘要）",
        ],
        "medium": [
            "细化评估标准，让 Agent 更清楚期望的输出风格",
            "增加参考答案的覆盖范围",
            "对关键字段添加输出校验",
        ],
    },
    "latency": {
        "low": [
            "减少不必要的工具调用链长度",
            "使用异步并发执行独立工具调用",
            "启用 LLM 流式响应减少用户感知延迟",
            "考虑使用更轻量的模型（fast tier）处理简单请求",
        ],
        "medium": [
            "优化数据加载，使用列式存储（Parquet）替代 CSV",
            "增加工具级缓存，对重复查询直接返回",
            "检查网络延迟是否为瓶颈",
        ],
    },
}


def _get_suggestions(dimension: str, score: float) -> list[str]:
    """Get improvement suggestions based on dimension and score."""
    level = "low" if score < 0.5 else "medium" if score < 0.8 else "high"
    dim_suggestions = _SUGGESTIONS.get(dimension, {})
    return dim_suggestions.get(level, ["当前表现良好，继续保持。"])


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class EvalReport:
    """Generate Markdown evaluation reports from EvalResult list.

    Usage:
        results = [evaluator.evaluate_agent_run(...), ...]
        report = EvalReport(results)
        markdown = report.to_markdown()
        report.save("eval_report.md")
    """

    def __init__(self, results: list[EvalResult]) -> None:
        self.results = results
        self.generated_at = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Markdown generation
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Generate the full Markdown report."""
        sections = [
            self._header(),
            self._overall_summary(),
            self._per_agent_details(),
            self._dimension_analysis(),
            self._improvement_suggestions(),
            self._footer(),
        ]
        return "\n\n".join(sections)

    def _header(self) -> str:
        return "\n".join([
            "# GrowthPilot Agent 评测报告",
            "",
            f"> 生成时间: {self.generated_at}",
            f"> 样本数量: {len(self.results)}",
        ])

    def _overall_summary(self) -> str:
        """Overall score summary table."""
        if not self.results:
            return "## 总体摘要\n\n无评测结果。"

        scores = [r.overall_score for r in self.results]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        # Per-dimension averages
        dims = ["task_completion", "tool_efficiency", "output_quality", "latency"]
        dim_avgs = {}
        for dim in dims:
            dim_scores = [r.dimensions[dim].score for r in self.results if dim in r.dimensions]
            dim_avgs[dim] = sum(dim_scores) / len(dim_scores) if dim_scores else 0.0

        dim_labels = {
            "task_completion": "任务完成度",
            "tool_efficiency": "工具效率",
            "output_quality": "输出质量",
            "latency": "响应延迟",
        }

        lines = [
            "## 总体摘要",
            "",
            f"| 指标 | 分数 | 等级 |",
            f"|------|------|------|",
            f"| **总体平均分** | {avg_score:.2f} | {self._grade(avg_score)} |",
            f"| 最低分 | {min_score:.2f} | {self._grade(min_score)} |",
            f"| 最高分 | {max_score:.2f} | {self._grade(max_score)} |",
            "",
            "### 各维度平均分",
            "",
            f"| 维度 | 平均分 | 等级 |",
            f"|------|--------|------|",
        ]
        for dim in dims:
            avg = dim_avgs.get(dim, 0.0)
            label = dim_labels.get(dim, dim)
            lines.append(f"| {label} | {avg:.2f} | {self._grade(avg)} |")

        return "\n".join(lines)

    def _per_agent_details(self) -> str:
        """Per-agent detailed results."""
        if not self.results:
            return ""

        lines = [
            "## 各 Agent 评测详情",
            "",
        ]

        dim_labels = {
            "task_completion": "任务完成度",
            "tool_efficiency": "工具效率",
            "output_quality": "输出质量",
            "latency": "响应延迟",
        }

        for i, r in enumerate(self.results, 1):
            lines.append(f"### {i}. {r.agent_name} (总分: {r.overall_score:.2f})")
            lines.append("")
            lines.append(f"**输入查询**: {r.input_query[:100]}{'...' if len(r.input_query) > 100 else ''}")
            lines.append("")
            lines.append("| 维度 | 分数 | 评估说明 |")
            lines.append("|------|------|----------|")

            for dim_name, dim_score in r.dimensions.items():
                label = dim_labels.get(dim_name, dim_name)
                reason_short = dim_score.reason[:80]
                if len(dim_score.reason) > 80:
                    reason_short += "..."
                lines.append(f"| {label} | {dim_score.score:.2f} | {reason_short} |")

            lines.append("")

        return "\n".join(lines)

    def _dimension_analysis(self) -> str:
        """Cross-cutting dimension analysis."""
        if not self.results:
            return ""

        dims = ["task_completion", "tool_efficiency", "output_quality", "latency"]
        dim_labels = {
            "task_completion": "任务完成度",
            "tool_efficiency": "工具效率",
            "output_quality": "输出质量",
            "latency": "响应延迟",
        }

        lines = [
            "## 维度分析",
            "",
        ]

        for dim in dims:
            dim_results = [(r.agent_name, r.dimensions[dim]) for r in self.results if dim in r.dimensions]
            if not dim_results:
                continue

            avg = sum(ds.score for _, ds in dim_results) / len(dim_results)
            best_agent, best_score = max(dim_results, key=lambda x: x[1].score)
            worst_agent, worst_score = min(dim_results, key=lambda x: x[1].score)

            lines.append(f"### {dim_labels.get(dim, dim)} (平均: {avg:.2f})")
            lines.append("")
            lines.append(f"- 最佳: {best_agent} ({best_score:.2f})")
            lines.append(f"- 最差: {worst_agent} ({worst_score:.2f})")

            # Show reasons
            for agent, ds in dim_results:
                if ds.score < 0.5:
                    lines.append(f"- ⚠️ {agent}: {ds.reason}")

            lines.append("")

        return "\n".join(lines)

    def _improvement_suggestions(self) -> str:
        """Generate actionable improvement suggestions."""
        if not self.results:
            return ""

        dims = ["task_completion", "tool_efficiency", "output_quality", "latency"]
        dim_labels = {
            "task_completion": "任务完成度",
            "tool_efficiency": "工具效率",
            "output_quality": "输出质量",
            "latency": "响应延迟",
        }

        lines = [
            "## 改进建议",
            "",
        ]

        # Collect all weak dimensions
        weak_dims: dict[str, list[tuple[str, float]]] = {}
        for r in self.results:
            for dim in dims:
                if dim in r.dimensions:
                    score = r.dimensions[dim].score
                    if score < 0.8:
                        weak_dims.setdefault(dim, []).append((r.agent_name, score))

        if not weak_dims:
            lines.append("所有维度评分均在 0.8 以上，当前表现良好！建议持续监控。")
        else:
            for dim in dims:
                if dim not in weak_dims:
                    continue
                label = dim_labels.get(dim, dim)
                lines.append(f"### {label}")
                lines.append("")

                for agent, score in weak_dims[dim]:
                    lines.append(f"**{agent}** (分数: {score:.2f})")
                    suggestions = _get_suggestions(dim, score)
                    for s in suggestions:
                        lines.append(f"- {s}")
                    lines.append("")

        return "\n".join(lines)

    def _footer(self) -> str:
        return "\n".join([
            "---",
            "",
            f"*报告由 GrowthPilot EvalReport 自动生成 @ {self.generated_at}*",
        ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grade(score: float) -> str:
        """Convert score to grade label."""
        if score >= 0.9:
            return "A (优秀)"
        if score >= 0.8:
            return "B (良好)"
        if score >= 0.7:
            return "C (中等)"
        if score >= 0.5:
            return "D (需改进)"
        return "F (不合格)"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the report to a Markdown file."""
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")
        logger.info("Evaluation report saved to %s", p)
