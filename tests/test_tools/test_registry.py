"""Tests for src.tools.registry - ToolRegistry and @register decorator."""

from __future__ import annotations

import pytest

from src.tools.registry import ToolRegistry, ToolMeta, register


# ---------------------------------------------------------------------------
# Sample tools for testing
# ---------------------------------------------------------------------------


class DummyTool:
    """A simple tool for testing."""

    def __init__(self, value: int = 0):
        self.value = value


class AnotherTool:
    """Another tool for testing."""

    def __init__(self, label: str = ""):
        self.label = label


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the ToolRegistry before and after each test."""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolRegistryRegister:
    """Tests for tool registration."""

    def test_register_and_create(self):
        """Register a tool and create an instance."""
        ToolRegistry.register_class("dummy_tool", DummyTool, category="test")
        instance = ToolRegistry.create("dummy_tool", value=42)
        assert isinstance(instance, DummyTool)
        assert instance.value == 42

    def test_register_decorator(self):
        """The @register decorator registers a tool class."""

        @register("decorated_tool", category="test", description="A test tool")
        class DecoratedTool:
            pass

        assert ToolRegistry.has("decorated_tool")
        instance = ToolRegistry.create("decorated_tool")
        assert isinstance(instance, DecoratedTool)

    def test_register_overwrites_existing(self):
        """Registering the same name overwrites the previous tool."""
        ToolRegistry.register_class("my_tool", DummyTool)

        class NewTool:
            pass

        ToolRegistry.register_class("my_tool", NewTool)
        instance = ToolRegistry.create("my_tool")
        assert isinstance(instance, NewTool)


class TestToolRegistryList:
    """Tests for listing and discovery."""

    def test_list_tools_empty(self):
        """Empty registry returns empty list."""
        assert ToolRegistry.list_tools() == []

    def test_list_tools_returns_sorted_names(self):
        """list_tools returns sorted tool names."""
        ToolRegistry.register_class("zebra_tool", DummyTool, category="test")
        ToolRegistry.register_class("alpha_tool", AnotherTool, category="test")

        tools = ToolRegistry.list_tools()
        assert tools == ["alpha_tool", "zebra_tool"]

    def test_list_tools_by_category(self):
        """Filter tools by category."""
        ToolRegistry.register_class("t1", DummyTool, category="prospect")
        ToolRegistry.register_class("t2", AnotherTool, category="conversion")
        ToolRegistry.register_class("t3", DummyTool, category="prospect")

        prospect_tools = ToolRegistry.list_tools_by_category("prospect")
        assert prospect_tools == ["t1", "t3"]

    def test_get_meta(self):
        """Get metadata for a registered tool."""
        ToolRegistry.register_class(
            "meta_tool", DummyTool,
            category="test", description="Meta test tool", version="2.0.0",
        )
        meta = ToolRegistry.get_meta("meta_tool")
        assert meta is not None
        assert meta.name == "meta_tool"
        assert meta.category == "test"
        assert meta.version == "2.0.0"


class TestToolRegistryCreate:
    """Tests for the factory method."""

    def test_unknown_tool_raises(self):
        """Creating an unregistered tool raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            ToolRegistry.create("nonexistent")

    def test_create_with_kwargs(self):
        """Create passes kwargs to the tool constructor."""
        ToolRegistry.register_class("kw_tool", AnotherTool)
        instance = ToolRegistry.create("kw_tool", label="hello")
        assert instance.label == "hello"


class TestToolRegistryMaintenance:
    """Tests for unregister, clear, has, get."""

    def test_unregister(self):
        ToolRegistry.register_class("to_remove", DummyTool)
        assert ToolRegistry.has("to_remove")
        ToolRegistry.unregister("to_remove")
        assert not ToolRegistry.has("to_remove")

    def test_has(self):
        assert not ToolRegistry.has("nope")
        ToolRegistry.register_class("yep", DummyTool)
        assert ToolRegistry.has("yep")

    def test_get(self):
        ToolRegistry.register_class("getter", DummyTool)
        cls = ToolRegistry.get("getter")
        assert cls is DummyTool

    def test_get_nonexistent(self):
        assert ToolRegistry.get("ghost") is None

    def test_summary(self):
        ToolRegistry.register_class("s1", DummyTool, category="test", description="First")
        ToolRegistry.register_class("s2", AnotherTool, category="test", description="Second")

        summary = ToolRegistry.summary()
        assert len(summary) == 2
        assert summary[0]["name"] == "s1"
        assert summary[1]["name"] == "s2"
