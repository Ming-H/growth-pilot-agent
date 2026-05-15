"""Database connection and session management."""
import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    """Get or create the async database engine."""
    global _engine
    if _engine is None:
        s = get_settings()
        _engine = create_async_engine(
            s.db_url,
            echo=s.db_echo,
            pool_size=s.db_pool_size,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
        )
    return _session_factory


def dispose_engine() -> None:
    """Dispose of the engine and session factory (for shutdown/testing)."""
    global _engine, _session_factory
    if _engine is not None:
        import asyncio
        try:
            asyncio.get_event_loop().run_until_complete(_engine.dispose())
        except RuntimeError:
            pass
    _engine = None
    _session_factory = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session."""
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize database tables.

    In demo mode, uses create_all() for convenience.
    In production, expects Alembic migrations (alembic upgrade head).
    """
    s = get_settings()
    if s.demo_mode:
        logger.warning("Demo mode: using create_all() for table creation")
        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    else:
        logger.info("Production mode: tables should be managed via Alembic migrations")
        # Still ensure tables exist (useful for first-run with empty DB)
        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
