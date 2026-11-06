"""Stage 1: migrations produce the expected schema, indexes and read-only role."""

from __future__ import annotations

import pytest
from scripts.seed_db import build_data
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

pytestmark = pytest.mark.integration


async def _indexes(engine: AsyncEngine) -> dict[str, str]:
    async with engine.connect() as c:
        rows = await c.execute(
            text("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'")
        )
        return {r[0]: r[1] for r in rows}


async def test_tables_and_foreign_keys(engine: AsyncEngine) -> None:
    async with engine.connect() as c:
        fks = (
            await c.execute(
                text(
                    """
                    SELECT tc.table_name, ccu.table_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.constraint_column_usage ccu
                      ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                    """
                )
            )
        ).all()
    assert {
        ("accounts", "clients"),
        ("trades", "accounts"),
        ("trades", "instruments"),
        ("chunks", "documents"),
    } <= set(map(tuple, fks))


async def test_vector_and_fulltext_indexes(engine: AsyncEngine) -> None:
    idx = await _indexes(engine)
    assert "USING hnsw (embedding vector_cosine_ops)" in idx["ix_chunks_embedding_hnsw"]
    assert "USING gin (tsv)" in idx["ix_chunks_tsv_gin"]
    assert "ix_trades_account_id_traded_at" in idx
    assert "ix_accounts_client_id" in idx


async def test_tsvector_is_generated(engine: AsyncEngine) -> None:
    async with engine.begin() as c:
        await c.execute(
            text(
                "INSERT INTO documents (id, source_path, title, doc_type, content_hash) "
                "VALUES ('t1', 'x', 't', 'faq', 'h')"
            )
        )
        await c.execute(
            text(
                "INSERT INTO chunks (document_id, chunk_index, content, token_count, "
                "content_hash, embedding) VALUES ('t1', 0, 'Wire transfers require approval',"
                " 4, 'h', array_fill(0.1, ARRAY[768])::vector)"
            )
        )
        tsv = await c.execute(text("SELECT tsv::text FROM chunks WHERE document_id='t1'"))
        assert "'approv'" in tsv.scalar_one()
        await c.execute(text("DELETE FROM documents WHERE id='t1'"))


async def test_readonly_role_cannot_write_or_read_rag_tables(readonly_url: str) -> None:
    ro = create_async_engine(readonly_url)
    try:
        async with ro.connect() as c:
            assert (await c.execute(text("SHOW statement_timeout"))).scalar_one() == "5s"
            await c.execute(text("SELECT count(*) FROM trades"))
        for stmt in ("DELETE FROM trades", "SELECT * FROM chunks", "CREATE TABLE x (a int)"):
            async with ro.connect() as c:
                with pytest.raises(DBAPIError):
                    await c.execute(text(stmt))
    finally:
        await ro.dispose()


def test_seed_data_is_deterministic() -> None:
    a = build_data(50, 500)
    b = build_data(50, 500)
    assert a == b
    _, clients, accounts, trades = a
    assert len(clients) == 50 and len(trades) == 500
    account_ids = {acc[0] for acc in accounts}
    assert all(t[1] in account_ids for t in trades)
    assert {t[3] for t in trades} == {"BUY", "SELL"}
