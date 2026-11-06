"""Stage 4: executor + introspector against Postgres as the read-only role."""

from __future__ import annotations

from collections.abc import AsyncIterator

import asyncpg
import pytest
from scripts.seed_db import build_data
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from enterprise_rag.sql.executor import SQLExecutionError, SQLExecutor
from enterprise_rag.sql.guard import GuardedSQL, SQLGuard
from enterprise_rag.sql.introspector import SchemaIntrospector

pytestmark = pytest.mark.integration
TABLES = ("clients", "accounts", "trades", "instruments")


@pytest.fixture(scope="module")
async def seeded(pg_url: str) -> None:
    instruments, clients, accounts, trades = build_data(40, 1500)
    conn = await asyncpg.connect(pg_url.replace("+asyncpg", ""))
    try:
        await conn.execute("TRUNCATE trades, accounts, clients, instruments CASCADE")
        await conn.copy_records_to_table("instruments", records=instruments)
        await conn.copy_records_to_table("clients", records=clients)
        await conn.copy_records_to_table("accounts", records=accounts)
        await conn.copy_records_to_table("trades", records=trades)
        await conn.execute("ANALYZE")
    finally:
        await conn.close()


@pytest.fixture(scope="module")
async def ro_engine(readonly_url: str, seeded: None) -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(readonly_url)
    yield eng
    await eng.dispose()


async def test_guarded_query_executes(ro_engine: AsyncEngine) -> None:
    guard = SQLGuard(TABLES, max_rows=10)
    res = await SQLExecutor(ro_engine).execute(guard.validate("SELECT id, price FROM trades"))
    assert res.columns == ["id", "price"]
    assert res.row_count == 10  # LIMIT enforced
    assert isinstance(res.rows[0][1], float)  # Decimal -> float


async def test_statement_timeout_enforced(ro_engine: AsyncEngine) -> None:
    # Bypass the guard on purpose: the executor's own timeout is the second layer.
    slow = GuardedSQL("SELECT pg_sleep(2)", frozenset(), False)
    with pytest.raises(SQLExecutionError, match="statement timeout"):
        await SQLExecutor(ro_engine, statement_timeout_ms=100).execute(slow)


async def test_read_only_even_if_guard_bypassed(ro_engine: AsyncEngine) -> None:
    write = GuardedSQL("DELETE FROM trades", frozenset(), False)
    with pytest.raises(SQLExecutionError, match="read-only"):
        await SQLExecutor(ro_engine).execute(write)


async def test_introspector_describes_keys_and_enums(ro_engine: AsyncEngine) -> None:
    intro = SchemaIntrospector(ro_engine, TABLES, notes=["-- note"])
    desc = await intro.describe()
    assert "TABLE trades (" in desc
    assert "account_id integer REFERENCES accounts.id" in desc
    assert "id integer PRIMARY KEY" in desc
    assert "'BUY', 'SELL'" in desc
    assert "region character varying  -- values: 'AMER'" in desc
    assert desc.endswith("-- note")
    assert "chunks" not in desc
    assert await intro.describe() is desc  # cached
    intro.invalidate()
    assert await intro.describe() == desc
