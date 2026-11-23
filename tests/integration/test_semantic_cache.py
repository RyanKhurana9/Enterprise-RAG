"""Stage 5: semantic cache against real Postgres + pgvector."""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np
import pytest
from sqlalchemy import text

from enterprise_rag.cache.response_cache import ResponseCache
from enterprise_rag.config import Settings
from enterprise_rag.db.engine import SessionFactory, create_engine, create_session_factory

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
async def sf(pg_url: str) -> AsyncIterator[SessionFactory]:
    engine = create_engine(pg_url, Settings(), application_name="test-cache")
    yield create_session_factory(engine)
    await engine.dispose()


def _unit(seed: int, base: np.ndarray | None = None, noise: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base.copy() if base is not None else rng.normal(size=768)
    if noise:
        v = v + rng.normal(scale=noise, size=768)
    return (v / np.linalg.norm(v)).astype(np.float32)


async def test_semantic_hit_threshold_and_number_guard(sf: SessionFactory) -> None:
    cache = ResponseCache(None, sf, semantic_threshold=0.9)
    await cache.clear()
    base = _unit(1)
    await cache.put("How many trades in 2024?", base, {"answer": "A"})

    near = _unit(2, base, noise=0.005)  # cosine ~0.99
    hit = await cache.get_semantic("how many trades were made in 2024", near)
    assert hit is not None and hit.kind == "semantic" and hit.response == {"answer": "A"}
    assert hit.similarity > 0.9 and hit.matched_query == "How many trades in 2024?"

    # Same embedding neighbourhood but a different year -> must not be served.
    assert await cache.get_semantic("how many trades in 2023", near) is None
    # Unrelated vector -> below threshold.
    assert await cache.get_semantic("How many trades in 2024?", _unit(3)) is None

    stats = await cache.stats()
    assert stats["semantic_entries"] == 1 and stats["semantic_hits"] == 1


async def test_upsert_and_eviction(sf: SessionFactory) -> None:
    cache = ResponseCache(None, sf, max_entries=3, evict_every=1)
    await cache.clear()
    for i in range(5):
        await cache.put(f"question {i}", _unit(10 + i), {"i": i})
    await cache.put("question 4", _unit(14), {"i": "updated"})  # same hash: upsert
    async with sf() as s:
        n = (await s.execute(text("SELECT count(*) FROM semantic_cache"))).scalar_one()
        latest = (
            await s.execute(
                text("SELECT response FROM semantic_cache WHERE query_text = 'question 4'")
            )
        ).scalar_one()
    assert n == 3 and latest == {"i": "updated"}
