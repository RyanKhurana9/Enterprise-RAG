"""Show the Postgres query plans for the retrieval queries (index usage + timings).

    python -m scripts.explain ["query text"] [--force-index]

Shows whether vector search uses the HNSW index, keyword search the GIN index, and the
semantic-cache lookup its own HNSW index. On a small corpus (a few hundred chunks) the
planner correctly prefers a sequential scan because it is cheaper; ``--force-index`` sets
``enable_seqscan = off`` to display the plan the indexes give at scale.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from enterprise_rag.cache.response_cache import LOOKUP_SQL
from enterprise_rag.config import get_settings
from enterprise_rag.container import Container
from enterprise_rag.retrieval.sql_retrievers import KEYWORD_SQL, VECTOR_SQL


async def explain(c: Container, query: str, *, force_index: bool = False) -> None:
    emb = await c.embedder.embed_query(query)
    plans = [
        ("vector (HNSW)", VECTOR_SQL, {"q": emb, "k": 20}),
        ("keyword (GIN tsvector)", KEYWORD_SQL, {"query": query, "k": 20}),
        ("semantic cache (HNSW)", LOOKUP_SQL, {"q": emb, "ttl": 86400}),
    ]
    for name, stmt, params in plans:
        async with c.session_factory() as s, s.begin():
            await s.execute(text(f"SET LOCAL hnsw.ef_search = {c.settings.hnsw_ef_search}"))
            if force_index:
                await s.execute(text("SET LOCAL enable_seqscan = off"))
            rows = await s.execute(
                text(f"EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) {stmt.text}"), params
            )
            print(f"\n=== {name} ===")
            for (line,) in rows:
                print(line)


async def main_async(query: str, force_index: bool) -> None:
    c = Container.build(get_settings())
    try:
        await explain(c, query, force_index=force_index)
    finally:
        await c.aclose()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query", nargs="?", default="What is the hotel cap in the travel policy?")
    p.add_argument("--force-index", action="store_true", help="disable sequential scans")
    args = p.parse_args()
    asyncio.run(main_async(args.query, args.force_index))


if __name__ == "__main__":
    main()
