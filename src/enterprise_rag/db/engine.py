"""Async engine / session factories with connection pooling and the pgvector codec."""

from __future__ import annotations

from typing import Any

from pgvector.asyncpg import register_vector
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from enterprise_rag.config import Settings


def create_engine(
    url: str, settings: Settings, *, application_name: str, vector: bool = True
) -> AsyncEngine:
    """Create a pooled asyncpg engine.

    * ``pool_pre_ping`` survives Postgres restarts.
    * ``application_name`` makes ``pg_stat_activity`` readable.
    * The pgvector *binary* codec is registered on every new connection, so embeddings are
      sent as numpy float32 arrays instead of 768-number text literals.
    """
    engine = create_async_engine(
        url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_s,
        pool_pre_ping=True,
        connect_args={"server_settings": {"application_name": application_name}},
    )
    if vector:
        register_vector_codec(engine)
    return engine


def register_vector_codec(engine: AsyncEngine) -> None:
    @event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        dbapi_connection.run_async(register_vector)


SessionFactory = async_sessionmaker[AsyncSession]


def create_session_factory(engine: AsyncEngine) -> SessionFactory:
    return async_sessionmaker(engine, expire_on_commit=False)
