"""Hybrid retrieval: run child retrievers concurrently and fuse with RRF."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from enterprise_rag.ingestion.embedder import Embedder, Vector
from enterprise_rag.retrieval.base import RetrievedChunk, Retriever
from enterprise_rag.retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        embedder: Embedder | None = None,
        rrf_k: int = 60,
        candidates_per_retriever: int = 20,
        weights: Sequence[float] | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("need at least one retriever")
        self._retrievers = list(retrievers)
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._candidates = candidates_per_retriever
        self._weights = weights

    async def retrieve(
        self, query: str, k: int, *, embedding: Vector | None = None
    ) -> list[RetrievedChunk]:
        if embedding is None and self._embedder is not None:
            embedding = await self._embedder.embed_query(query)  # embed once, share it
        n = max(k, self._candidates)
        # Each child uses its own pooled connection, so the two SQL queries run in parallel.
        results = await asyncio.gather(
            *(r.retrieve(query, n, embedding=embedding) for r in self._retrievers)
        )
        by_key: dict[str, RetrievedChunk] = {}
        for result in results:
            for c in result:
                by_key.setdefault(c.key, c)
        fused = reciprocal_rank_fusion(
            [[c.key for c in result] for result in results],
            k=self._rrf_k,
            weights=self._weights,
        )
        return [by_key[key].with_score(score) for key, score in fused[:k]]
