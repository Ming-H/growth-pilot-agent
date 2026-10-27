"""Tests for the KPI reconciliation gate (对账闸门)."""

from __future__ import annotations

from src.core.kpi_gate import extract_claims, reconcile, render_appendix, run_gate

SNAPSHOT = {
    "conversion_rate": 0.08,
    "expected_roi": 2.1,
    "total_users": 500,
    "churn_high_risk_count": 5.0,
}


class TestExtractClaims:
    def test_kpi_context_number_is_claim(self):
        report = "本月转化率达到 8%，表现良好"
        claims = extract_claims(report)
        assert any(c["raw"] == "8%" for c in claims)

    def test_number_without_kpi_context_ignored(self):
        report = "会议安排在 3 月 15 日，共 7 项议程"
        assert extract_claims(report) == []

    def test_plain_ratio_matches_snapshot(self):
        report = "转化率为 0.08 的基线"
        claims = extract_claims(report)
        assert any(c["raw"] == "0.08" for c in claims)

    def test_multiplier_form(self):
        report = "补贴 ROI 达到 2.1x"
        claims = extract_claims(report)
        assert any(c["raw"].startswith("2.1") for c in claims)

    def test_thousands_separator(self):
        report = "覆盖用户数 1,250,000 人"
        claims = extract_claims(report)
        assert any("1,250,000" in c["raw"] for c in claims)

    def test_dates_not_claims(self):
        report = "2024/10 报告期，转化率 8%"
        claims = extract_claims(report)
        assert all("2024" not in c["raw"] for c in claims)


class TestReconcile:
    def test_clean_report_passes(self):
        report = "转化率 8%，ROI 2.1x，用户数 500"
        result = reconcile(report, SNAPSHOT)
        assert result["ok"] is True
        assert not result["violations"]

    def test_fabricated_number_trips(self):
        report = "转化率 99% 创历史新高"
        result = reconcile(report, SNAPSHOT)
        assert result["ok"] is False
        assert any(c["raw"] == "99%" for c in result["violations"])

    def test_percent_matches_ratio_ground_truth(self):
        """8% reconciles against snapshot value 0.08."""
        result = reconcile("转化率 8%", {"conversion_rate": 0.08})
        assert result["ok"] is True

    def test_verified_ledger_extends_ground_truth(self):
        result = reconcile("召回率 35%", {}, verified_values=[0.35])
        assert result["ok"] is True

    def test_rounding_tolerance(self):
        result = reconcile("转化率 8.0%", {"conversion_rate": 0.0801})
        assert result["ok"] is True


class TestRunGate:
    def _state(self, report=None, **kw):
        return {"kpi_snapshot": SNAPSHOT, "final_report": report, **kw}

    def test_mode_off_skips(self):
        update = run_gate(
            "转化率 99%", self._state(), mode="off", threshold=0.0, deterministic_report="tmpl"
        )
        assert update["kpi_gate_result"].get("skipped") is True

    def test_mode_on_degrades_to_template(self):
        update = run_gate(
            "转化率 99%",
            self._state(),
            mode="on",
            threshold=0.0,
            deterministic_report="# 确定性报告\n\n转化率 8%",
        )
        assert update["final_report"].startswith("# 确定性报告")
        assert "kpi_gate:degraded" in update["degradations"]
        assert "对账附录" in update["final_report"]

    def test_mode_annotate_keeps_report(self):
        update = run_gate(
            "本月转化率 99%",
            self._state(),
            mode="annotate",
            threshold=0.0,
            deterministic_report="tmpl",
        )
        assert update["final_report"].startswith("本月转化率 99%")
        assert "kpi_gate:annotated" in update["degradations"]

    def test_clean_pass_returns_no_degradation(self):
        update = run_gate(
            "转化率 8%", self._state(), mode="on", threshold=0.0, deterministic_report="tmpl"
        )
        assert "degradations" not in update

    def test_threshold_tolerates_fraction(self):
        # 1 violation out of 3 claims, threshold 0.5 → tolerated (annotated only)
        report = "转化率 8%，ROI 2.1，流失率 77%"
        update = run_gate(
            report, self._state(), mode="on", threshold=0.5, deterministic_report="tmpl"
        )
        assert "kpi_gate:degraded" not in update.get("degradations", [])


class TestAppendix:
    def test_renders_violation_table(self):
        result = reconcile("转化率 99%", SNAPSHOT)
        appendix = render_appendix(result)
        assert "对账附录" in appendix
        assert "99%" in appendix
