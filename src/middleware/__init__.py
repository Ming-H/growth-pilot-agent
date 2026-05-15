"""Middleware base class and implementations."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AgentMiddleware(ABC):
    """Base class for agent middleware. Intercepts model and tool calls."""

    @abstractmethod
    async def wrap_model_call(
        self, request: dict[str, Any], handler: Callable
    ) -> dict[str, Any]:
        """Intercept model calls."""
        ...

    async def wrap_tool_call(
        self, request: dict[str, Any], handler: Callable
    ) -> dict[str, Any]:
        """Intercept tool calls. Default: pass through."""
        return await handler(request)


class ToolErrorHandlingMiddleware(AgentMiddleware):
    """Catch and normalize tool call errors to prevent workflow crashes."""

    async def wrap_model_call(self, request: dict[str, Any], handler: Callable) -> dict[str, Any]:
        return await handler(request)

    async def wrap_tool_call(self, request: dict[str, Any], handler: Callable) -> dict[str, Any]:
        try:
            return await handler(request)
        except Exception as e:
            tool_name = request.get("name", "unknown")
            logger.warning("Tool %s failed: %s", tool_name, e)
            return {
                "content": f"[{tool_name}] 执行出错: {type(e).__name__}: {e}",
                "status": "error",
            }


class LoggingMiddleware(AgentMiddleware):
    """Log model and tool calls with timing."""

    def __init__(self, level: str = "INFO") -> None:
        self.level = level

    async def wrap_model_call(self, request: dict[str, Any], handler: Callable) -> dict[str, Any]:
        start = time.time()
        response = await handler(request)
        elapsed = time.time() - start
        logger.log(getattr(logging, self.level), "Model call completed in %.2fs", elapsed)
        return response

    async def wrap_tool_call(self, request: dict[str, Any], handler: Callable) -> dict[str, Any]:
        tool_name = request.get("name", "unknown")
        start = time.time()
        response = await handler(request)
        elapsed = time.time() - start
        logger.log(getattr(logging, self.level), "Tool %s completed in %.2fs", tool_name, elapsed)
        return response


class RetryMiddleware(AgentMiddleware):
    """Retry model calls on failure with exponential backoff."""

    def __init__(self, max_retries: int = 3, backoff_base: float = 2.0) -> None:
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    async def wrap_model_call(self, request: dict[str, Any], handler: Callable) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = self.backoff_base ** attempt
                    logger.warning(
                        "Model call failed (attempt %d), retrying in %.1fs: %s",
                        attempt + 1, wait, e,
                    )
                    await asyncio.sleep(wait)
        raise last_error  # type: ignore[misc]

    async def wrap_tool_call(self, request: dict[str, Any], handler: Callable) -> dict[str, Any]:
        return await handler(request)


def build_middleware_stack(
    log_level: str = "INFO",
    max_retries: int | None = None,
) -> list[AgentMiddleware]:
    """Build the middleware stack with default ordering.

    When *max_retries* is ``None``, reads from ``Settings.max_retries``.
    """
    if max_retries is None:
        from src.core.config import get_settings
        max_retries = get_settings().max_retries
    return [
        ToolErrorHandlingMiddleware(),
        RetryMiddleware(max_retries=max_retries),
        LoggingMiddleware(level=log_level),
    ]
