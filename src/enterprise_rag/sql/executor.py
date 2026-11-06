"""Executes guarded SQL on the read-only connection pool."""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from enterprise_rag.sql.guard import GuardedSQL


class SQLExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SQLResult:
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    elapsed_ms: float


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    return value


class SQLExecutor:
    """Runs one statement per READ ONLY transaction with a hard timeout.

    ``engine`` must be bound to the ``rag_readonly`` role; the per-transaction settings
    repeat the role defaults so safety does not depend on role configuration alone.
    """

    def __init__(self, engine: AsyncEngine, *, statement_timeout_ms: int = 5000) -> None:
        self._engine = engine
        self._timeout_ms = int(statement_timeout_ms)

    async def execute(self, guarded: GuardedSQL) -> SQLResult:
        t0 = time.perf_counter()
        try:
            async with self._engine.connect() as conn, conn.begin():
                await conn.execute(text("SET TRANSACTION READ ONLY"))
                await conn.execute(text(f"SET LOCAL statement_timeout = {self._timeout_ms}"))
                result = await conn.execute(text(guarded.sql))
                columns = list(result.keys())
                rows = [[to_jsonable(v) for v in row] for row in result.fetchall()]
        except Exception as e:  # surface DB errors (incl. timeout) to the repair loop
            msg = str(getattr(e, "orig", e)).splitlines()[0]
            raise SQLExecutionError(msg) from e
        return SQLResult(columns, rows, len(rows), (time.perf_counter() - t0) * 1000)
