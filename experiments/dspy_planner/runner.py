"""DSPy 优化规划节点（agent_plan_node）：把手写规划提示词变成可编译的 Signature。

设计（详见同目录 README.md）：
· 优化对象 —— agent 模式的规划节点：LLM 把问题拆解为带专家指派的子目标，
  路由质量直接由它决定（任务级基准的 route 维）；
· 候选通道 —— DSPy 模块的输出先经**生产校验器**清洗（与线上同一道闸，
  只删不造），再与 ``builtin_tasks`` 的 ``expected_experts`` 真值比对；
· 对照组 —— 生产规则路由（``plan:rule`` 兜底同款，零成本）作为基线，
  优化前后各评一轮，三方对比（规则 / 手写提示词 / 编译产物）；
· 通道 —— 复用 GPA 的 ``GPA_LLM_*`` 配置（``src.core.config``），
  ``model_type="chat"`` 显式指定（兼容端点未实现 /responses，dspy 3.5 实测）。

用法（仓库根目录）：
  uv run python experiments/dspy_planner/runner.py            # 冒烟档
  uv run python experiments/dspy_planner/runner.py --full     # 完整 MIPROv2
  uv run python experiments/dspy_planner/runner.py --eval-compiled  # 只评编译产物
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metric as planner_metric  # noqa: E402

from src.core.config import get_settings  # noqa: E402
from src.evals.tasks import builtin_tasks  # noqa: E402

HERE = Path(__file__).resolve().parent
COMPILED = HERE / "compiled" / "planner.json"
FULL = "--full" in sys.argv


def build_signature():
    import dspy

    class GrowthPlanner(dspy.Signature):
        """你是增长分析规划器：把用户问题拆解为至多 5 个子目标，每个子目标
        指定一个专家（prospect/conversion/subsidy/retention/ad），超纲问题
        （与用户增长无关）应输出空清单。"""

        question: str = dspy.InputField(desc="用户提出的增长分析问题")
        budget: str = dspy.InputField(desc="预算（元），未指定则为空")
        subgoals_json: str = dspy.OutputField(
            desc='JSON: {"subgoals": [{"expert": "<key>", "question": "<该专家要回答的问题>"}]}；'
            '与增长无关的问题输出 {"subgoals": []}'
        )

    return dspy.Predict(GrowthPlanner)


def configure_lm():
    """复用 GPA 通道配置；model_type="chat" 必须显式（兼容端点 404 坑）。"""
    import dspy

    s = get_settings()
    lm = dspy.LM(
        f"openai/{s.llm_model}",
        model_type="chat",
        api_base=str(s.llm_base_url),
        api_key=s.llm_api_key or None,
        temperature=0,
        max_tokens=1024,
    )
    dspy.configure(lm=lm)
    return lm


def _trainset(tasks):
    import dspy

    return [
        dspy.Example(question=t.query, budget=str(t.budget) if t.budget else "").with_inputs(
            "question", "budget"
        )
        for t in tasks
    ]


def evaluate(planner, tasks, tag: str) -> list[dict]:
    rows = []
    for t in tasks:
        t0 = time.perf_counter()
        try:
            pred = planner(question=t.query, budget=str(t.budget) if t.budget else "")
            subs = planner_metric.parse_subgoals(pred.subgoals_json, t.query, t.budget)
        except Exception:
            subs = []
        r = planner_metric.score_task(t, subs)
        r["ms"] = round((time.perf_counter() - t0) * 1000)
        rows.append(r)
        print(
            f"  [{tag}] {t.id:<26} expect={r['expected']} got={r['got']} "
            f"score={r['score']} ({r['ms']}ms)",
            flush=True,
        )
    mean = sum(r["score"] for r in rows) / max(1, len(rows))
    print(f"  [{tag}] 均分 {mean:.3f}", flush=True)
    return rows


def main() -> None:
    import dspy

    tasks = builtin_tasks()
    print(f"实验配置：{'完整' if FULL else '冒烟'} · {len(tasks)} 题", flush=True)

    # 1) 规则路由基线（生产 plan:rule 兜底同款，零成本）
    print("\n[1/3] 规则路由基线", flush=True)
    rule_rows = []
    for t in tasks:
        got = planner_metric.rule_baseline_experts(t.query)
        r = planner_metric.score_task(t, [{"expert": e, "question": t.query} for e in got])
        rule_rows.append(r)
        print(f"  [rule] {t.id:<26} got={r['got']} score={r['score']}", flush=True)
    rule_mean = sum(r["score"] for r in rule_rows) / len(rule_rows)
    print(f"  [rule] 均分 {rule_mean:.3f}", flush=True)

    configure_lm()
    planner = build_signature()
    base_rows: list[dict] = []
    base_mean = None

    if "--eval-compiled" in sys.argv:
        if not COMPILED.exists():
            sys.exit("未找到 compiled/planner.json，先跑 --full")
        planner.load(str(COMPILED))
        print("\n[3/3] 编译产物评测", flush=True)
        opt_rows = evaluate(planner, tasks, "opt")
    else:
        # 2) 优化前（DSPy Predict = 手写提示词语义等价物）
        print("\n[2/3] 优化前评测（未编译 Predict）", flush=True)
        base_rows = evaluate(planner, tasks, "base")
        base_mean = sum(r["score"] for r in base_rows) / len(base_rows)

        # 3) MIPROv2 编译（instruction-only；num_threads=1 串行）
        print("\n[3/3] MIPROv2 编译中…", flush=True)

        def metric(example, pred, trace=None):
            t = next(x for x in tasks if x.query == example.question)
            subs = planner_metric.parse_subgoals(pred.subgoals_json, t.query, t.budget)
            return planner_metric.score_task(t, subs)["score"]

        if FULL:
            opt = dspy.MIPROv2(metric=metric, auto="light", num_threads=1)
            planner = opt.compile(
                planner, trainset=_trainset(tasks), max_bootstrapped_demos=0, max_labeled_demos=0
            )
        else:
            opt = dspy.MIPROv2(
                metric=metric, auto=None, num_candidates=2, init_temperature=0.3, num_threads=1
            )
            planner = opt.compile(
                planner,
                trainset=_trainset(tasks),
                minibatch_size=4,
                num_trials=2,
                max_bootstrapped_demos=0,
                max_labeled_demos=0,
            )
        COMPILED.parent.mkdir(parents=True, exist_ok=True)
        planner.save(str(COMPILED))
        opt_rows = evaluate(planner, tasks, "opt")

    opt_mean = sum(r["score"] for r in opt_rows) / len(opt_rows)

    print("\n════════ 结果对比 ════════")
    print(f"  规则路由基线 {rule_mean:.3f}")
    if base_mean is not None:
        print(f"  优化前(Predict) {base_mean:.3f}")
    print(f"  优化后(编译)   {opt_mean:.3f}")

    report = {
        "mode": "full" if FULL else "smoke",
        "n_tasks": len(tasks),
        "rule_baseline": {"mean": rule_mean, "rows": rule_rows},
        "optimized": {"mean": opt_mean, "rows": opt_rows},
    }
    if base_mean is not None:
        report["pre_optimization"] = {"mean": base_mean, "rows": base_rows}
    out = HERE / ("report-full.json" if FULL else "report-smoke.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告落盘 {out}")


if __name__ == "__main__":
    main()
