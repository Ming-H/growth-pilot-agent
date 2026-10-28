"""FastAPI application factory, lifespan, and exception handlers."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core import __version__
from src.core.config import get_settings
from src.core.observability import setup_telemetry
from src.db.database import init_db
from src.web.routers.analyses import AnalysisError
from src.web.routers.analyses import router as analyses_router
from src.web.routers.auth import router as auth_router
from src.web.routers.memory import router as memory_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    settings = get_settings()

    if not settings.demo_mode:
        if settings.jwt_secret == "change-me-in-production":
            raise ValueError(
                "FATAL: Default JWT secret detected in non-demo mode. "
                "Set GPA_JWT_SECRET to a secure random string."
            )
        if settings.channel == "remote" and not settings.llm_api_key:
            raise ValueError(
                "FATAL: Remote LLM channel without an API key. "
                "Set GPA_LLM_API_KEY, point GPA_LLM_BASE_URL at a local endpoint,"
                " or enable GPA_DEMO_MODE=true."
            )

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )
    logger.info(
        "GrowthPilot API starting (version=%s, mode=%s, channel=%s)",
        app.version,
        settings.orchestration_mode,
        settings.channel,
    )

    setup_telemetry(settings.otel_service_name, enabled=settings.otel_enabled)

    try:
        await init_db()
        logger.info("Database tables initialized")
    except Exception as exc:
        logger.warning("Database init failed (tables may already exist): %s", exc)

    yield
    logger.info("GrowthPilot API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="GrowthPilot Agent API",
        version=__version__,
        description="Reconciliation-gated multi-agent user growth platform",
        lifespan=lifespan,
    )

    settings = get_settings()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting (only if slowapi is available)
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        from slowapi.util import get_remote_address

        limiter = Limiter(key_func=get_remote_address)
        application.state.limiter = limiter
        application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    except ImportError:  # pragma: no cover
        logger.info("slowapi not installed — rate limiting disabled")

    application.include_router(auth_router)
    application.include_router(analyses_router)
    application.include_router(memory_router)

    @application.get("/api/v1/health", tags=["health"])
    async def health_check() -> dict:
        return {"status": "ok", "version": __version__}

    @application.exception_handler(AnalysisError)
    async def analysis_error_handler(request: Request, exc: AnalysisError) -> JSONResponse:
        return JSONResponse(
            status_code=500, content={"success": False, "error": exc.message, "detail": exc.detail}
        )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "error": exc.detail, "status_code": exc.status_code},
        )

    @application.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        settings = get_settings()
        detail = str(exc) if settings.demo_mode else "Internal server error"
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error", "detail": detail},
        )

    return application


app = create_app()
