"""Tool registry for dynamic tool discovery, registration, and creation.

Provides:
- @register decorator for auto-registering tool classes
- ToolRegistry.create() factory for instantiation by name
- ToolRegistry.list_tools() for discovery
- Metadata support (category, description)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool metadata model
# ---------------------------------------------------------------------------

class ToolMeta(BaseModel):
    """Metadata associated with a registered tool."""

    name: str
    category: str = ""
    description: str = ""
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Centralized registry for growth tools.

    Tools can be registered via the ``@register`` decorator or the
    ``register()`` classmethod. Once registered, tools are instantiated
    via ``create(name, **kwargs)``.

    Usage::

        @register("feature_engine", category="prospect")
        class FeatureEngine:
            ...

        engine = ToolRegistry.create("feature_engine")
        all_tools = ToolRegistry.list_tools()
    """

    _tools: dict[str, Type] = {}
    _meta: dict[str, ToolMeta] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        *,
        category: str = "",
        description: str = "",
        version: str = "1.0.0",
    ) -> Callable:
        """Decorator to register a tool class.

        Args:
            name: Unique tool identifier.
            category: Tool category (prospect/conversion/subsidy/retention/ad/common).
            description: Human-readable tool description.
            version: Tool version string.
        """
        def decorator(tool_cls: Type) -> Type:
            if name in cls._tools:
                logger.warning("Tool '%s' already registered, overwriting", name)
            cls._tools[name] = tool_cls
            cls._meta[name] = ToolMeta(
                name=name,
                category=category,
                description=description or tool_cls.__doc__ or "",
                version=version,
            )
            logger.debug("Registered tool: %s (category=%s)", name, category)
            return tool_cls
        return decorator

    @classmethod
    def register_class(
        cls,
        name: str,
        tool_cls: Type,
        *,
        category: str = "",
        description: str = "",
        version: str = "1.0.0",
    ) -> None:
        """Programmatically register a tool class (non-decorator usage)."""
        cls.register(name, category=category, description=description, version=version)(tool_cls)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        """Create a tool instance by name.

        Args:
            name: Registered tool name.
            **kwargs: Arguments forwarded to the tool constructor.

        Raises:
            ValueError: If the tool name is not registered.
        """
        if name not in cls._tools:
            raise ValueError(
                f"Unknown tool: '{name}'. "
                f"Available: {cls.list_tools()}"
            )
        return cls._tools[name](**kwargs)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def list_tools(cls) -> list[str]:
        """List all registered tool names, sorted alphabetically."""
        return sorted(cls._tools.keys())

    @classmethod
    def list_tools_by_category(cls, category: str) -> list[str]:
        """List tool names filtered by category."""
        return sorted(
            name for name, meta in cls._meta.items()
            if meta.category == category
        )

    @classmethod
    def get_meta(cls, name: str) -> ToolMeta | None:
        """Get metadata for a registered tool."""
        return cls._meta.get(name)

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a tool is registered."""
        return name in cls._tools

    @classmethod
    def get(cls, name: str) -> Type | None:
        """Get the tool class (not instance) by name."""
        return cls._tools.get(name)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a tool from the registry."""
        cls._tools.pop(name, None)
        cls._meta.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools."""
        cls._tools.clear()
        cls._meta.clear()

    @classmethod
    def summary(cls) -> list[dict[str, Any]]:
        """Get a summary of all registered tools with metadata."""
        result = []
        for name in cls.list_tools():
            meta = cls._meta.get(name, ToolMeta(name=name))
            result.append({
                "name": name,
                "category": meta.category,
                "description": meta.description[:80],
                "version": meta.version,
            })
        return result


# ---------------------------------------------------------------------------
# Convenience alias for the decorator
# ---------------------------------------------------------------------------

# Module-level decorator shorthand: @register("name", category="...")
register = ToolRegistry.register
