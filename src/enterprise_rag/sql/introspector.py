"""Reads the warehouse schema from information_schema / pg_stats and renders a compact,
LLM-friendly description. The result is cached (TTL) because the schema changes only with
migrations and introspection costs several catalog queries."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

COLUMNS_SQL = text(
    """
    SELECT c.table_name, c.column_name, c.data_type
    FROM information_schema.columns c
    WHERE c.table_schema = 'public' AND c.table_name = ANY(:tables)
    ORDER BY c.table_name, c.ordinal_position
    """
)
# pg_catalog rather than information_schema: information_schema.table_constraints only
# lists tables on which the role has a privilege *other than* SELECT, so the read-only
# role would see no keys at all.
PK_FK_SQL = text(
    """
    SELECT CASE con.contype WHEN 'p' THEN 'PRIMARY KEY' ELSE 'FOREIGN KEY' END
               AS constraint_type,
           cl.relname AS table_name, att.attname AS column_name,
           rcl.relname AS ref_table, ratt.attname AS ref_column
    FROM pg_constraint con
    JOIN pg_class cl ON cl.oid = con.conrelid
    JOIN pg_namespace ns ON ns.oid = cl.relnamespace
    CROSS JOIN LATERAL unnest(con.conkey, con.confkey) AS k(attnum, refattnum)
    JOIN pg_attribute att ON att.attrelid = con.conrelid AND att.attnum = k.attnum
    LEFT JOIN pg_class rcl ON rcl.oid = con.confrelid
    LEFT JOIN pg_attribute ratt ON ratt.attrelid = con.confrelid AND ratt.attnum = k.refattnum
    WHERE ns.nspname = 'public' AND cl.relname = ANY(:tables) AND con.contype IN ('p', 'f')
    """
)
# Low-cardinality columns: list their values so the model writes region = 'EMEA' rather
# than region = 'Europe'. pg_stats (filled by ANALYZE) nominates candidates cheaply;
# negative n_distinct is a fraction of the row count. Exact values are then fetched.
ENUM_CANDIDATES_SQL = text(
    """
    SELECT s.tablename, s.attname
    FROM pg_stats s
    JOIN pg_namespace n ON n.nspname = s.schemaname
    JOIN pg_class c ON c.relname = s.tablename AND c.relnamespace = n.oid
    WHERE s.schemaname = 'public' AND s.tablename = ANY(:tables)
      AND CASE WHEN s.n_distinct < 0 THEN -s.n_distinct * c.reltuples
               ELSE s.n_distinct END <= :max_distinct
    """
)
TEXT_TYPES = frozenset({"character varying", "text", "character"})


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class SchemaIntrospector:
    def __init__(
        self,
        engine: AsyncEngine,
        tables: Sequence[str],
        *,
        ttl_s: float = 3600.0,
        max_enum_values: int = 30,
        notes: Sequence[str] = (),
    ) -> None:
        self._engine = engine
        self._tables = list(tables)
        self._ttl = ttl_s
        self._max_enum = max_enum_values
        self._notes = list(notes)
        self._cached: str = ""
        self._cached_at = 0.0
        self._lock = asyncio.Lock()

    async def describe(self) -> str:
        if self._cached and time.monotonic() - self._cached_at < self._ttl:
            return self._cached
        async with self._lock:  # one introspection even under a burst of requests
            if not self._cached or time.monotonic() - self._cached_at >= self._ttl:
                self._cached = await self._introspect()
                self._cached_at = time.monotonic()
        return self._cached

    def invalidate(self) -> None:
        self._cached = ""

    async def _introspect(self) -> str:
        params = {"tables": self._tables}
        async with self._engine.connect() as conn:
            cols = (await conn.execute(COLUMNS_SQL, params)).all()
            keys = (await conn.execute(PK_FK_SQL, params)).all()
            text_cols = {(r.table_name, r.column_name) for r in cols if r.data_type in TEXT_TYPES}
            candidates = (
                await conn.execute(ENUM_CANDIDATES_SQL, params | {"max_distinct": self._max_enum})
            ).all()
            values: dict[tuple[str, str], list[str]] = {}
            for table, column in candidates:
                if (table, column) not in text_cols:
                    continue
                # Identifiers come from the catalog and are quoted; no user input here.
                rows = await conn.execute(
                    text(
                        f"SELECT DISTINCT {_ident(column)} FROM {_ident(table)} "
                        f"WHERE {_ident(column)} IS NOT NULL ORDER BY 1 LIMIT :n"
                    ),
                    {"n": self._max_enum + 1},
                )
                vals = [str(v) for (v,) in rows]
                if len(vals) <= self._max_enum:
                    values[(table, column)] = vals

        pks = {(r.table_name, r.column_name) for r in keys if r.constraint_type == "PRIMARY KEY"}
        fks = {
            (r.table_name, r.column_name): f"{r.ref_table}.{r.ref_column}"
            for r in keys
            if r.constraint_type == "FOREIGN KEY"
        }

        lines: list[str] = []
        current = None
        for r in cols:
            if r.table_name != current:
                if current is not None:
                    lines.append(")")
                lines.append(f"TABLE {r.table_name} (")
                current = r.table_name
            col = f"  {r.column_name} {r.data_type}"
            if (r.table_name, r.column_name) in pks:
                col += " PRIMARY KEY"
            if ref := fks.get((r.table_name, r.column_name)):
                col += f" REFERENCES {ref}"
            if enum_vals := values.get((r.table_name, r.column_name)):
                col += "  -- values: " + ", ".join(f"'{v}'" for v in enum_vals)
            lines.append(col)
        if current is not None:
            lines.append(")")
        lines += self._notes
        return "\n".join(lines)
