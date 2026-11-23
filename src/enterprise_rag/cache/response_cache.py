"""Two-tier response cache.

1. **Exact** (Redis): key = sha256 of the normalised query. Sub-millisecond, TTL-bounded.
2. **Semantic** (Postgres ``semantic_cache`` + pgvector HNSW): the nearest previously
   answered query by cosine similarity; a hit requires ``similarity >= threshold``.

Semantic caching has one classic failure: "trades in 2023" and "trades in 2024" embed
almost identically but have different answers. So a semantic hit also requires the two
queries to contain the *same numbers and upper-case codes* (years, limits, tickers,
regions). The cache is strictly best-effort: any Redis/Postgres error is logged and treated
as a miss, so the cache can never fail a request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from redis.asyncio import Redis
from sqlalchemy import text

from enterprise_rag.db.engine import SessionFactory
from enterprise_rag.ingestion.embedder import Vector

log = logging.getLogger(__name__)

EXACT_PREFIX = "rag:exact:"

LOOKUP_SQL = text(
    """
    SELECT id, query_text, response, 1 - (embedding <=> :q) AS similarity
    FROM semantic_cache
    WHERE created_at > now() - make_interval(secs => :ttl)
    ORDER BY embedding <=> :q
    LIMIT 1
    """
)
TOUCH_SQL = text(
    "UPDATE semantic_cache SET hit_count = hit_count + 1, last_accessed_at = now() WHERE id = :id"
)
UPSERT_SQL = text(
    """
    INSERT INTO semantic_cache (query_text, query_hash, embedding, response)
    VALUES (:query_text, :query_hash, :embedding, CAST(:response AS jsonb))
    ON CONFLICT (query_hash) DO UPDATE
        SET response = EXCLUDED.response, embedding = EXCLUDED.embedding,
            created_at = now(), last_accessed_at = now()
    """
)
# Keep the newest/most recently used ``max_entries`` rows (LRU-ish eviction).
EVICT_SQL = text(
    """
    DELETE FROM semantic_cache WHERE id IN (
        SELECT id FROM semantic_cache ORDER BY last_accessed_at DESC OFFSET :max_entries
    )
    """
)

_SPACE = re.compile(r"\s+")
_KEY_TOKENS = re.compile(r"\d+(?:[.,]\d+)*|\b[A-Z]{2,6}\b")


def normalize_query(query: str) -> str:
    return _SPACE.sub(" ", query).strip().lower().rstrip("?.! ")


def query_hash(query: str) -> str:
    return hashlib.sha256(normalize_query(query).encode()).hexdigest()


def key_tokens(query: str) -> frozenset[str]:
    """Numbers and upper-case codes (years, amounts, tickers, regions) in the query."""
    return frozenset(t.replace(",", "") for t in _KEY_TOKENS.findall(query))


@dataclass(frozen=True, slots=True)
class CacheHit:
    kind: Literal["exact", "semantic"]
    response: dict[str, Any]
    similarity: float = 1.0
    matched_query: str = ""


class ResponseCache:
    def __init__(
        self,
        redis: Redis | None,
        session_factory: SessionFactory | None,
        *,
        exact_ttl_s: int = 3600,
        semantic_threshold: float = 0.92,
        semantic_ttl_s: int = 24 * 3600,
        max_entries: int = 10_000,
        evict_every: int = 100,
    ) -> None:
        self._redis = redis
        self._sf = session_factory
        self._exact_ttl = exact_ttl_s
        self.threshold = semantic_threshold
        self._semantic_ttl = semantic_ttl_s
        self._max_entries = max_entries
        self._evict_every = evict_every
        self._writes = 0

    # ---------------------------------------------------------------------- exact ----
    async def get_exact(self, query: str) -> CacheHit | None:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(EXACT_PREFIX + query_hash(query))
        except Exception as e:
            log.warning("exact cache get failed: %r", e)
            return None
        if raw is None:
            return None
        return CacheHit("exact", json.loads(raw))

    # ------------------------------------------------------------------- semantic ----
    async def get_semantic(self, query: str, embedding: Vector) -> CacheHit | None:
        if self._sf is None:
            return None
        try:
            async with self._sf() as s, s.begin():
                row = (
                    await s.execute(LOOKUP_SQL, {"q": embedding, "ttl": self._semantic_ttl})
                ).first()
                if row is None or float(row.similarity) < self.threshold:
                    return None
                if key_tokens(row.query_text) != key_tokens(query):
                    log.info("semantic near-miss rejected: %r vs %r", query, row.query_text)
                    return None
                await s.execute(TOUCH_SQL, {"id": row.id})
        except Exception as e:
            log.warning("semantic cache lookup failed: %r", e)
            return None
        response = row.response if isinstance(row.response, dict) else json.loads(row.response)
        return CacheHit("semantic", response, float(row.similarity), row.query_text)

    async def get(self, query: str, embedding: Vector | None) -> CacheHit | None:
        if (hit := await self.get_exact(query)) is not None:
            return hit
        if embedding is None:
            return None
        return await self.get_semantic(query, embedding)

    # ----------------------------------------------------------------------- write ----
    async def put(self, query: str, embedding: Vector | None, response: dict[str, Any]) -> None:
        payload = json.dumps(response, default=str)
        h = query_hash(query)
        if self._redis is not None:
            try:
                await self._redis.set(EXACT_PREFIX + h, payload, ex=self._exact_ttl)
            except Exception as e:
                log.warning("exact cache put failed: %r", e)
        if self._sf is None or embedding is None:
            return
        try:
            async with self._sf() as s, s.begin():
                await s.execute(
                    UPSERT_SQL,
                    {
                        "query_text": query,
                        "query_hash": h,
                        "embedding": embedding,
                        "response": payload,
                    },
                )
                self._writes += 1
                if self._writes % self._evict_every == 0:
                    await s.execute(EVICT_SQL, {"max_entries": self._max_entries})
        except Exception as e:
            log.warning("semantic cache put failed: %r", e)

    async def clear(self) -> int:
        removed = 0
        if self._redis is not None:
            try:
                keys = [k async for k in self._redis.scan_iter(match=EXACT_PREFIX + "*")]
                if keys:
                    removed += int(await self._redis.delete(*keys))
            except Exception as e:
                log.warning("exact cache clear failed: %r", e)
        if self._sf is not None:
            async with self._sf() as s, s.begin():
                res = await s.execute(text("DELETE FROM semantic_cache"))
                removed += int(getattr(res, "rowcount", 0) or 0)
        return removed

    async def stats(self) -> dict[str, int]:
        out = {"exact_entries": 0, "semantic_entries": 0, "semantic_hits": 0}
        if self._redis is not None:
            try:
                out["exact_entries"] = len(
                    [k async for k in self._redis.scan_iter(match=EXACT_PREFIX + "*")]
                )
            except Exception as e:
                log.warning("exact cache stats failed: %r", e)
        if self._sf is not None:
            async with self._sf() as s:
                row = (
                    await s.execute(
                        text("SELECT count(*), coalesce(sum(hit_count), 0) FROM semantic_cache")
                    )
                ).one()
                out["semantic_entries"], out["semantic_hits"] = int(row[0]), int(row[1])
        return out
