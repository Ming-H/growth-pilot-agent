"""Tests for src.core.models - Pydantic structured models."""
from __future__ import annotations

import pytest

from src.core.models import (
    AgentResult,
    ConversionResult,
    LLMAnalysis,
    KpiSnapshot,
    ProspectResult,
    SegmentInfo,
    SubsidyResult,
    RetentionResult,
    AdResult,
    BidResult,
    ChurnRisk,
)


class TestAgentResult:
    """Tests for the base AgentResult model."""

    def test_default_values(self):
        r = AgentResult()
        assert r.success is True
        assert r.errors == []
        assert r.agent_name == ""
        assert r.timestamp != ""

    def test_custom_values(self):
        r = AgentResult(success=False, errors=["err1"], agent_name="test")
        assert r.success is False
        assert r.errors == ["err1"]
        assert r.agent_name == "test"


class TestLLMAnalysis:
    """Tests for LLMAnalysis model."""

    def test_default_values(self):
        a = LLMAnalysis()
        assert a.summary == ""
        assert a.confidence == 0.0
        assert a.raw_response == ""
        assert a.extra == {}

    def test_confidence_bounds(self):
        a = LLMAnalysis(confidence=0.75)
        assert a.confidence == 0.75

        with pytest.raises(Exception):
            LLMAnalysis(confidence=1.5)

        with pytest.raises(Exception):
            LLMAnalysis(confidence=-0.1)


class TestProspectResult:
    """Tests for ProspectResult model."""

    def test_prospect_result_creation(self):
        r = ProspectResult(user_count=500)
        assert r.agent_name == "prospect"
        assert r.user_count == 500
        assert r.success is True
        assert isinstance(r.analysis, LLMAnalysis)

    def test_with_segments(self):
        r = ProspectResult(
            segment_summary={"high_intent": SegmentInfo(count=100, ratio=0.2)},
            rfm_result_count=3,
        )
        assert r.segment_summary["high_intent"].count == 100


class TestConversionResult:
    def test_creation(self):
        r = ConversionResult(funnel_result={"step1": 0.5})
        assert r.agent_name == "conversion"
        assert r.funnel_result["step1"] == 0.5


class TestSubsidyResult:
    def test_creation(self):
        r = SubsidyResult(expected_roi=3.5, confidence=0.8)
        assert r.agent_name == "subsidy"
        assert r.expected_roi == 3.5


class TestRetentionResult:
    def test_creation(self):
        r = RetentionResult(churn_risk=ChurnRisk(high_risk_ratio=0.1))
        assert r.agent_name == "retention"
        assert r.churn_risk.high_risk_ratio == 0.1


class TestAdResult:
    def test_creation(self):
        r = AdResult(
            bid_result=BidResult(original_bid=10.0, optimized_bid=8.5),
            expected_cpa=45.0,
        )
        assert r.agent_name == "ad"
        assert r.bid_result.optimized_bid == 8.5


class TestKpiSnapshot:
    def test_default_values(self):
        kpi = KpiSnapshot()
        assert kpi.total_users == 0
        assert kpi.scope == "full"

    def test_custom_values(self):
        kpi = KpiSnapshot(total_users=1000, conversion_rate=0.15, budget=50000)
        assert kpi.total_users == 1000
        assert kpi.conversion_rate == 0.15


class TestInvalidData:
    """Test Pydantic validation rejects invalid data."""

    def test_invalid_confidence(self):
        with pytest.raises(Exception):
            LLMAnalysis(confidence=5.0)

    def test_churn_risk_no_validation(self):
        cr = ChurnRisk(high_risk_ratio=1.5)
        assert cr.high_risk_ratio == 1.5

    def test_segment_info_invalid_ratio(self):
        with pytest.raises(Exception):
            SegmentInfo(ratio=2.0)
