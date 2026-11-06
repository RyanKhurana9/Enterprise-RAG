from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

ROOT = Path(__file__).resolve().parents[1]


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    """Session-wide pgvector Postgres via testcontainers, migrated to head with Alembic."""
    if not _docker_available():
        pytest.skip("Docker not available for testcontainers")
    from alembic import command
    from alembic.config import Config
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer(
        "pgvector/pgvector:pg16", username="rag", password="rag", dbname="rag", driver="asyncpg"
    ) as pg:
        url = pg.get_connection_url()
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT / "migrations"))
        cfg.set_main_option("sqlalchemy.url", url)
        os.environ["RAG_READONLY_PASSWORD"] = "readonly"
        command.upgrade(cfg, "head")
        yield url


@pytest.fixture(scope="session")
def readonly_url(pg_url: str) -> str:
    return pg_url.replace("rag:rag@", "rag_readonly:readonly@", 1)


@pytest.fixture(scope="session")
async def engine(pg_url: str) -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(pg_url, pool_size=5)
    yield eng
    await eng.dispose()
