"""Tests for src.memory.manager - MemoryManager with TF-IDF search."""

from __future__ import annotations

import json
import time

import pytest

from src.memory.manager import MemoryManager


class TestMemoryManagerStore:
    """Tests for storing entries."""

    def test_store_returns_id(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        entry_id = mm.store(
            run_id="run_001",
            query="货运增长分析",
            scope="full",
            results_summary={"user_count": 100},
        )
        assert isinstance(entry_id, str)
        assert len(entry_id) == 32  # uuid hex

    def test_store_persists_to_disk(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        mm.store("run_001", "test query", "prospect", "summary text")
        mm2 = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        assert mm2.count() == 1

    def test_store_multiple_entries(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        mm.store("run_001", "q1", "prospect", "s1")
        mm.store("run_002", "q2", "conversion", "s2")
        mm.store("run_003", "q3", "retention", {"data": "s3"})
        assert mm.count() == 3


class TestMemoryManagerSearch:
    """Tests for TF-IDF search."""

    def test_search_finds_relevant(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        mm.store("run_001", "货运转化率分析", "conversion", "转化率从3%提升到5%")
        mm.store("run_002", "用户流失预测", "retention", "高流失风险用户占比8%")

        results = mm.search("转化率")
        assert len(results) >= 1
        assert results[0]["query"] == "货运转化率分析"

    def test_search_returns_empty_for_no_entries(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        results = mm.search("anything")
        assert results == []

    def test_search_returns_empty_for_empty_query(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        mm.store("run_001", "test", "full", "summary")
        results = mm.search("")
        assert results == []

    def test_search_respects_limit(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        for i in range(10):
            mm.store(f"run_{i:03d}", "货运增长", "full", f"summary {i}")

        results = mm.search("货运", limit=3)
        assert len(results) <= 3

    def test_search_includes_relevance_score(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        mm.store("run_001", "货运增长分析", "full", "详细分析结果")

        results = mm.search("货运")
        assert len(results) >= 1
        assert "_relevance_score" in results[0]
        assert results[0]["_relevance_score"] > 0


class TestMemoryManagerGetRecent:
    """Tests for get_recent."""

    def test_get_recent_returns_newest_first(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        mm.store("run_001", "first query", "full", "s1")
        time.sleep(0.01)
        mm.store("run_002", "second query", "full", "s2")

        recent = mm.get_recent(limit=2)
        assert len(recent) == 2
        assert recent[0]["query"] == "second query"

    def test_get_recent_respects_limit(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        for i in range(5):
            mm.store(f"run_{i:03d}", f"query {i}", "full", f"s{i}")

        recent = mm.get_recent(limit=2)
        assert len(recent) == 2


class TestMemoryManagerBuildContext:
    """Tests for build_context."""

    def test_build_context_with_matching_memory(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        mm.store("run_001", "货运增长", "full", {"result": "positive"})

        context = mm.build_context("货运")
        assert "历史分析记忆" in context
        assert "货运增长" in context

    def test_build_context_empty_memory(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        context = mm.build_context("some query")
        assert context == ""

    def test_build_context_empty_query(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        mm.store("run_001", "test", "full", "summary")
        context = mm.build_context("")
        assert context == ""


class TestMemoryManagerClear:
    """Tests for clear and count."""

    def test_clear_removes_all(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        mm.store("run_001", "q1", "full", "s1")
        mm.store("run_002", "q2", "full", "s2")
        assert mm.count() == 2

        removed = mm.clear()
        assert removed == 2
        assert mm.count() == 0

    def test_clear_persists(self, tmp_memory_dir):
        mm = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        mm.store("run_001", "q1", "full", "s1")
        mm.clear()

        mm2 = MemoryManager(base_path=str(tmp_memory_dir), org_id="test-org")
        assert mm2.count() == 0
