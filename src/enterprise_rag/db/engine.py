"""Async engine / session factories with connection pooling."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from enterprise_rag.config import Settings


def create_engine(url: str, settings: Settings, *, application_name: str) -> AsyncEngine:
    """Create a pooled asyncpg engine.

    ``pool_pre_ping`` protects against Postgres restarts; ``application_name`` makes
    pg_stat_activity readable when debugging.
    """
    return create_async_engine(
        url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_s,
        pool_pre_ping=True,
        connect_args={"server_settings": {"application_name": application_name}},
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker:  # type: ignore[type-arg]
    return async_sessionmaker(engine, expire_on_commit=False)
