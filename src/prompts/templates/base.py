"""PromptTemplate base class with Jinja2 rendering and structured XML tags.

Design principles:
- Jinja2 rendering for dynamic context injection
- XML tag structure for LLM reasoning guidance (<analysis>, <strategy>, <output>)
- User-growth domain-specific defaults
- No external dependencies beyond jinja2 (already in pyproject.toml)
"""
from __future__ import annotations

import logging
from typing import Any

from jinja2 import BaseLoader, Environment, StrictUndefined

logger = logging.getLogger(__name__)

# Shared Jinja2 environment — strict mode catches undefined variables early
_JINJA_ENV = Environment(
    loader=BaseLoader(),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


class PromptTemplate:
    """Base prompt template for GrowthPilot agents.

    Subclasses define:
    - ``TEMPLATE``: Jinja2 template string
    - ``DEFAULT_CONTEXT``: dict of default values for template variables

    Usage::

        prompt = ProspectPrompt().render(
            user_count=500,
            intent_metrics={"auc": 0.82},
            season="春季",
        )
    """

    TEMPLATE: str = ""
    DEFAULT_CONTEXT: dict[str, Any] = {}

    def __init__(self) -> None:
        self._template = _JINJA_ENV.from_string(self.TEMPLATE)

    def render(self, **kwargs: Any) -> str:
        """Render the template with the given context variables.

        Merges DEFAULT_CONTEXT with provided kwargs (kwargs take precedence).
        Automatically injects ``role_definition`` and ``business_context``
        from ``@property`` methods when subclasses define them.
        """
        # Auto-inject property-based context that templates reference
        auto_ctx: dict[str, Any] = {}
        if hasattr(type(self), "role_definition") and isinstance(
            getattr(type(self), "role_definition"), property
        ):
            auto_ctx["role_definition"] = self.role_definition
        if hasattr(type(self), "business_context") and isinstance(
            getattr(type(self), "business_context"), property
        ):
            auto_ctx["business_context"] = self.business_context

        ctx = {**self.DEFAULT_CONTEXT, **auto_ctx, **kwargs}
        try:
            return self._template.render(**ctx)
        except Exception as exc:
            logger.warning("PromptTemplate render failed: %s", exc)
            # Fallback: return the raw template with simple string substitution
            result = self.TEMPLATE
            for k, v in ctx.items():
                result = result.replace("{{ " + k + " }}", str(v))
            return result

    # ------------------------------------------------------------------
    # XML structured sections — helper methods for subclass composition
    # ------------------------------------------------------------------

    @staticmethod
    def wrap_xml(tag: str, content: str) -> str:
        """Wrap content in XML tags: <tag>content</tag>."""
        return f"<{tag}>\n{content}\n</{tag}>"

    @staticmethod
    def role_definition(
        agent_name: str,
        capabilities: list[str],
    ) -> str:
        """Generate a standard role definition section."""
        caps = "\n".join(f"- {c}" for c in capabilities)
        return f"你是 GrowthPilot {agent_name}。\n\n## 核心能力\n{caps}"

    @staticmethod
    def business_context(platform: str = "平台", extra: str = "") -> str:
        """Generate the standard business context paragraph.

        This is the shared platform context used by all agents.
        """
        base = (
            f"你服务于一个日活 5000 万的{platform}。平台有两条获客渠道：\n"
            "- 端内渠道（主渠道，80%+）：从平台用户池识别转化意向用户\n"
            "- 域外渠道（辅助，<20%）：通过 RTA/OCPX 在抖音/快手投放广告\n"
            "北极星指标：月活跃用户数（MAO）。"
        )
        if extra:
            return f"{base}\n{extra}"
        return base

    @staticmethod
    def output_format_json(schema: dict[str, str]) -> str:
        """Generate <output> section with a JSON schema description."""
        lines = [f'  "{k}": "{v}"' for k, v in schema.items()]
        body = ",\n".join(lines)
        return PromptTemplate.wrap_xml(
            "output",
            "{\n" + body + "\n}",
        )
