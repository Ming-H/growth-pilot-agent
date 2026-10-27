"""KPI 对账闸门 (reconciliation gate) — deterministic, zero LLM.

Tally's core discipline adapted to growth analytics: **LLM 组织语言，
确定性闸门拥有事实**。The final report may only carry numbers that trace
back to tool-computed KPIs (``kpi_snapshot`` ∪ L1 verified facts); claims
that cannot be reconciled trip the gate.

How it works:

1. **Claim extraction** (pure regex): scan the report for numbers/percent-
   ages that appear near a KPI lexicon word (转化率/ROI/AUC/CPA/流失/留存/
   补贴/意向/用户数/客单价/ROI/churn/retention/conversion…). Numbers with
   no KPI context (dates, section numbering, prices) are ignored.
2. **Reconciliation**: each claim must match some known value after
   inverse formatting — ``8%`` ≡ ``0.08`` (± display rounding), ``0.08``
   ≡ raw ratio, ``12,345`` ≡ 12345, ``2.1x`` ≡ 2.1. Matching against ANY
   snapshot/ledger value with tolerance at the displayed precision.
3. **Verdict**:
   - ``on``       — violations ⇒ replace the report with the deterministic
                    template + a reconciliation appendix (safe failure),
                    marker ``kpi_gate:degraded``
   - ``annotate`` — keep the report, append the discrepancy table,
                    marker ``kpi_gate:annotated``
   - ``off``      — skip

Tolerated-violation fraction (``Settings.kpi_gate_threshold``) dials the
strictness from 0.0 (any violation trips) to 1.0 (never trips).
"""

from __future__ import annotations

import re
from typing import Any

# ── KPI lexicon: a number is a *claim* only when a KPI word is near ──────
_KPI_WORDS = (
    # Chinese
    "转化率|转换率|流失|留存|召回|意向|用户数|客单价|复购|渗透|点击|曝光|"
    "ROI|投产比|回报率|ROI|AUC|auc|CPA|CPM|CPC|CTR|CVR|GMV|LTV|ARPU|"
    "补贴|预算|成本|毛利|增长|提升|下降|占比|比例|份额|命中率|准确率"
    # English words are covered by the alternation above (ROI/AUC/…)
)

# number patterns: percentages, multipliers, thousands, plain decimals
_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
_PCT = rf"{_NUM}\s*%"
_MULT = rf"{_NUM}\s*[xX×倍]"
_PLAIN = _NUM

# a claim = KPI word within `window` chars of the number (same line)
_WINDOW = 24

_LINE_CLAIM_RE = re.compile(
    rf"(?:(?P<w1>{_KPI_WORDS}).{{0,{_WINDOW}}}?)?(?P<n1>{_PCT}|{_MULT}|{_PLAIN})"
    rf"(?:.{{0,{_WINDOW}}}?(?P<w2>{_KPI_WORDS}))?"
)

_FRACTION_RE = re.compile(r"\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?")  # dates/ratios like 2024/10


def _parse_number(raw: str) -> float | None:
    try:
        return float(
            raw.replace(",", "")
            .replace("x", "")
            .replace("X", "")
            .replace("×", "")
            .replace("倍", "")
            .replace("%", "")
            .strip()
        )
    except ValueError:
        return None


def extract_claims(report: str) -> list[dict[str, Any]]:
    """Extract KPI-contextual numeric claims from a report."""
    claims: list[dict[str, Any]] = []
    for line_no, line in enumerate(report.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("|") and "---" in stripped:
            continue
        # Skip pure table separators and headings
        if re.fullmatch(r"[-|: ]+", stripped):
            continue
        # skip fractions like dates 2024/10 (both sides numeric)
        for m in _LINE_CLAIM_RE.finditer(stripped):
            num_raw = (m.group("n1") or "").strip()
            if not num_raw:
                continue
            has_word = bool(m.group("w1") or m.group("w2"))
            # table rows: "| high_intent | 40 | 32.0% |" — the row label may
            # itself carry KPI semantics; treat any table-row number as a
            # claim only when a KPI word appears anywhere in the line.
            if not has_word and not (stripped.startswith("|") and re.search(_KPI_WORDS, stripped)):
                continue
            if _FRACTION_RE.search(stripped[max(0, m.start() - 2) : m.end() + 2]):
                continue
            value = _parse_number(num_raw)
            if value is None:
                continue
            is_pct = num_raw.rstrip().endswith("%")
            claims.append(
                {
                    "raw": num_raw,
                    "value": value,
                    "is_pct": is_pct,
                    "line": line_no,
                    "context": stripped[:80],
                }
            )
    return claims


def _values_match(claim: dict[str, Any], known: float) -> bool:
    """Match a claim against a known value with display-precision tolerance."""
    v = claim["value"]
    raw = claim["raw"]
    decimals = 2
    if "." in raw:
        decimals = len(raw.split(".")[-1].rstrip("%xX×倍"))
    tol = 0.5 * (10**-decimals)
    if claim["is_pct"]:
        # claim 8% matches known 0.08 or known 8.0
        return abs(v - known) <= tol or abs(v / 100.0 - known) <= tol / 100.0 + 1e-9
    return abs(v - known) <= tol


def reconcile(
    report: str,
    kpi_snapshot: dict[str, Any],
    verified_values: list[float] | None = None,
) -> dict[str, Any]:
    """Return {'claims': n, 'violations': [...], 'ok': bool}.

    ``ok`` is True when every extracted claim traces to at least one known
    KPI value (snapshot ∪ ledger).
    """
    known: list[float] = []
    for v in (kpi_snapshot or {}).values():
        try:
            known.append(float(v))
        except (TypeError, ValueError):
            continue
    known.extend(x for x in (verified_values or []) if x is not None)

    claims = extract_claims(report)
    violations: list[dict[str, Any]] = []
    for c in claims:
        if not any(_values_match(c, k) for k in known):
            violations.append(c)

    return {
        "claims": len(claims),
        "violations": violations,
        "ok": not violations,
        "known_values": len(known),
    }


def render_appendix(result: dict[str, Any]) -> str:
    """Deterministic reconciliation appendix (claimed vs traceable)."""
    lines = [
        "## 对账附录（KPI Reconciliation Appendix）",
        "",
        f"- 报告中 KPI 语境数字断言: {result['claims']} 处",
        f"- 无法回溯到工具产出 KPI 的断言: {len(result['violations'])} 处",
    ]
    if result["violations"]:
        lines.append("")
        lines.append("| 行号 | 断言数字 | 上下文 |")
        lines.append("|------|----------|--------|")
        for v in result["violations"][:20]:
            ctx = v["context"].replace("|", "\\|")
            lines.append(f"| {v['line']} | {v['raw']} | {ctx} |")
    lines.append("")
    lines.append(
        "*本报告由确定性对账闸门生成：所有数字均可回溯到专家工具产出的 KPI。*"
        if not result["violations"]
        else "*以下问题断言已如上标注；请以 KPI 快照为准。*"
    )
    return "\n".join(lines)


def run_gate(
    final_report: str,
    state: dict[str, Any],
    *,
    mode: str,
    threshold: float,
    deterministic_report: str,
    verified_values: list[float] | None = None,
) -> dict[str, Any]:
    """Run the gate; returns a state update dict.

    Args:
        final_report: the LLM (or agent) produced report.
        state: graph state (for kpi_snapshot / degradations).
        mode: on | annotate | off.
        threshold: tolerated fraction of untraceable claims.
        deterministic_report: template render used for degradation.
        verified_values: extra ground-truth values (L1 ledger).
    """
    if mode == "off":
        return {"kpi_gate_result": {"skipped": True}}

    snapshot = state.get("kpi_snapshot") or {}
    result = reconcile(final_report, snapshot, verified_values)
    tol_ok = (
        result["claims"] == 0 or len(result["violations"]) / max(result["claims"], 1) <= threshold
    )

    update: dict[str, Any] = {"kpi_gate_result": result}
    if result["ok"] or tol_ok:
        if result["violations"]:
            update["final_report"] = final_report + "\n\n" + render_appendix(result)
            update["degradations"] = ["kpi_gate:annotated"]
        return update

    if mode == "annotate":
        update["final_report"] = final_report + "\n\n" + render_appendix(result)
        update["degradations"] = ["kpi_gate:annotated"]
        return update

    # mode == "on": strict — replace with the deterministic report + appendix
    update["final_report"] = deterministic_report + "\n\n" + render_appendix(result)
    update["degradations"] = ["kpi_gate:degraded"]
    return update
