"""API-test isolation for the rate limiter.

The limiter is module-global and keyed by remote address (always
127.0.0.1 under TestClient/ASGITransport) — without resetting, hits
accumulate ACROSS tests and later tests would 429 randomly. Default:
disabled + reset; the dedicated rate-limit test re-enables it.
"""

from __future__ import annotations

import pytest

from src.web.rate_limit import HAS_SLOWAPI, limiter


@pytest.fixture(autouse=True)
def _isolated_limiter():
    if HAS_SLOWAPI and limiter is not None:
        limiter.reset()
        limiter.enabled = False
    yield
    if HAS_SLOWAPI and limiter is not None:
        limiter.enabled = True
        limiter.reset()
