"""Shared rate limiter — ONE instance for the app middleware AND routes.

C1: the Limiter used to be created in two places (app.py registered only
the exception handler; analyses.py defined a ``_limit_analyze`` decorator
that was never applied to anything) and ``SlowAPIMiddleware`` was never
mounted — rate limiting existed in name only. One shared instance lives
here, is mounted in ``create_app``, and decorates the abuse-prone routes
(login/register brute force, expensive analyze runs).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    HAS_SLOWAPI = True
except ImportError:  # pragma: no cover
    limiter = None  # type: ignore[assignment]
    HAS_SLOWAPI = False
    logger.info("slowapi not installed — rate limiting disabled")


def limit_or_pass(limit_str: str) -> Callable[[Any], Any]:
    """Return ``limiter.limit(limit_str)`` or an identity decorator."""
    if HAS_SLOWAPI and limiter is not None:
        return limiter.limit(limit_str)  # type: ignore[union-attr]

    def _identity(fn: Any) -> Any:
        return fn

    return _identity
