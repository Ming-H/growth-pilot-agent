"""Tests for src.evals.dataset - EvalDataset and EvalSample."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evals.dataset import EvalDataset, EvalSample


class TestEvalSample:
    """Tests for EvalSample model."""

    def test_default_values(self):
        s = EvalSample()
        assert s.id == ""
        assert s.input_query == ""
        assert s.scope == "full"
        assert s.expected_keys == []
        assert s.reference_answer == ""
        assert s.agent_name == ""
        assert s.metadata == {}

    def test_custom_values(self):
        s = EvalSample(
            id="test_001",
            input_query="test query",
            scope="prospect",
            expected_keys=["user_count"],
            agent_name="prospect",
        )
        assert s.id == "test_001"
        assert s.scope == "prospect"


class TestBuiltinDataset:
    """Tests for the built-in dataset."""

    def test_builtin_dataset_loads(self):
        ds = EvalDataset.from_builtin()
        assert len(ds) > 0

    def test_builtin_samples_have_ids(self):
        ds = EvalDataset.from_builtin()
        for sample in ds:
            assert sample.id != "", f"Sample missing id: {sample}"

    def test_builtin_samples_have_queries(self):
        ds = EvalDataset.from_builtin()
        for sample in ds:
            assert sample.input_query != ""

    def test_builtin_samples_have_agent_names(self):
        ds = EvalDataset.from_builtin()
        agents = {s.agent_name for s in ds}
        # Should cover the main agents
        assert "prospect" in agents
        assert "conversion" in agents
        assert "orchestrator" in agents


class TestFilterByAgent:
    """Tests for filter_by_agent."""

    def test_filter_by_agent(self):
        ds = EvalDataset.from_builtin()
        prospect_samples = ds.filter_by_agent("prospect")
        assert len(prospect_samples) > 0
        for s in prospect_samples:
            assert s.agent_name == "prospect"

    def test_filter_by_nonexistent_agent(self):
        ds = EvalDataset.from_builtin()
        results = ds.filter_by_agent("nonexistent")
        assert results == []

    def test_filter_by_scope(self):
        ds = EvalDataset.from_builtin()
        prospect_scope = ds.filter_by_scope("prospect")
        assert len(prospect_scope) > 0


class TestSampleCount:
    """Tests for sample counting and statistics."""

    def test_sample_count(self):
        ds = EvalDataset.from_builtin()
        assert ds.__len__() > 0
        assert len(ds.samples) == len(ds)

    def test_summary(self):
        ds = EvalDataset.from_builtin()
        summary = ds.summary()
        assert "total_samples" in summary
        assert summary["total_samples"] > 0
        assert "agents" in summary
        assert "scopes" in summary

    def test_get_by_id(self):
        ds = EvalDataset.from_builtin()
        # Get first sample's id
        first_id = ds.samples[0].id
        found = ds.get(first_id)
        assert found is not None
        assert found.id == first_id

    def test_get_nonexistent_id(self):
        ds = EvalDataset.from_builtin()
        assert ds.get("nonexistent_id") is None


class TestFromJson:
    """Tests for loading from JSON files."""

    def test_load_list_format(self, tmp_path):
        data = [
            {"id": "s1", "input_query": "q1", "agent_name": "prospect"},
            {"id": "s2", "input_query": "q2", "agent_name": "conversion"},
        ]
        f = tmp_path / "samples.json"
        f.write_text(json.dumps(data))
        ds = EvalDataset.from_json(f)
        assert len(ds) == 2

    def test_load_dict_format(self, tmp_path):
        data = {
            "samples": [
                {"id": "s1", "input_query": "q1"},
            ]
        }
        f = tmp_path / "samples.json"
        f.write_text(json.dumps(data))
        ds = EvalDataset.from_json(f)
        assert len(ds) == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            EvalDataset.from_json("/nonexistent/path.json")

    def test_invalid_format(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text('"just a string"')
        with pytest.raises(ValueError, match="must be a list"):
            EvalDataset.from_json(f)


class TestExportJson:
    """Tests for dataset export."""

    def test_to_json(self, tmp_path):
        ds = EvalDataset.from_builtin()
        out_path = tmp_path / "export.json"
        ds.to_json(out_path)
        assert out_path.exists()

        # Re-load and verify
        ds2 = EvalDataset.from_json(out_path)
        assert len(ds2) == len(ds)
