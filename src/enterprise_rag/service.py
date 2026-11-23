"""RAGService: the end-to-end question-answering flow.

    query -> exact cache -> embed -> semantic cache -> route
      docs:   hybrid retrieve -> rerank -> generate with citations
      sql:    Text2SQL (guard, read-only exec) -> summarise
      hybrid: both concurrently -> generate from both

Every stage is timed (``timings_ms`` in the response and Prometheus histograms). Failures
in one source degrade the answer instead of failing the request where possible.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Literal

from enterprise_rag.cache.response_cache import ResponseCache
from enterprise_rag.generation.answer import AnswerGenerator, GeneratedAnswer
from enterprise_rag.ingestion.embedder import Embedder, Vector
from enterprise_rag.metrics import QUERIES, QUERY_LATENCY, STAGE_LATENCY
from enterprise_rag.retrieval.base import RetrievedChunk, Retriever
from enterprise_rag.retrieval.reranker import Reranker
from enterprise_rag.routing.router import QueryRouter, Route, RouteDecision
from enterprise_rag.schemas import (
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    RouteInfo,
    Source,
    SQLPayload,
)
from enterprise_rag.sql.text2sql import Text2SQL, Text2SQLResult

log = logging.getLogger(__name__)


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @asynccontextmanager
    async def stage(self, name: str) -> AsyncIterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.ms[name] = round(self.ms.get(name, 0.0) + dt * 1000, 1)
            STAGE_LATENCY.labels(name).observe(dt)


def to_sources(
    chunks: Sequence[RetrievedChunk],
    cited: Sequence[RetrievedChunk] = (),
    *,
    snippet_chars: int = 700,
) -> list[Source]:
    cited_keys = {c.key for c in cited}
    return [
        Source(
            n=i,
            key=c.key,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            title=c.title,
            score=round(c.score, 4),
            snippet=c.content[:snippet_chars],
            cited=c.key in cited_keys,
        )
        for i, c in enumerate(chunks, 1)
    ]


def to_sql_payload(res: Text2SQLResult, *, max_rows: int = 200) -> SQLPayload:
    r = res.result
    return SQLPayload(
        sql=res.sql,
        columns=r.columns if r else [],
        rows=r.rows[:max_rows] if r else [],
        row_count=r.row_count if r else 0,
        attempts=res.attempts,
        errors=res.errors,
        ok=res.ok,
    )


class RAGService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        router: QueryRouter,
        retrievers: dict[str, Retriever],
        reranker: Reranker,
        text2sql: Text2SQL,
        generator: AnswerGenerator,
        cache: ResponseCache | None = None,
        candidates: int = 20,
        top_k: int = 5,
    ) -> None:
        if "hybrid" not in retrievers:
            raise ValueError("retrievers must include 'hybrid'")
        self._embedder = embedder
        self._router = router
        self._retrievers = retrievers
        self._reranker = reranker
        self._text2sql = text2sql
        self._gen = generator
        self._cache = cache
        self._candidates = candidates
        self._top_k = top_k

    # ------------------------------------------------------------------ retrieval ----
    async def _retrieve_docs(
        self,
        query: str,
        embedding: Vector | None,
        timer: _Timer,
        *,
        top_k: int,
        mode: str = "hybrid",
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        retriever = self._retrievers[mode]
        n = max(self._candidates, top_k) if rerank else top_k
        async with timer.stage("retrieve"):
            candidates = await retriever.retrieve(query, n, embedding=embedding)
        if not rerank:
            return candidates[:top_k]
        async with timer.stage("rerank"):
            return await self._reranker.rerank(query, candidates, top_k)

    async def retrieve(self, req: RetrieveRequest) -> RetrieveResponse:
        timer = _Timer()
        if req.mode not in self._retrievers:
            raise ValueError(f"unknown retriever mode: {req.mode}")
        embedding = None
        if req.mode != "keyword":
            async with timer.stage("embed"):
                embedding = await self._embedder.embed_query(req.query)
        chunks = await self._retrieve_docs(
            req.query, embedding, timer, top_k=req.k, mode=req.mode, rerank=req.rerank
        )
        return RetrieveResponse(
            query=req.query,
            mode=req.mode,
            reranked=req.rerank,
            sources=to_sources(chunks),
            timings_ms=timer.ms,
        )

    # ---------------------------------------------------------------------- answer ----
    async def answer(self, req: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()
        timer = _Timer()
        query = req.query.strip()
        use_cache = req.use_cache and self._cache is not None
        top_k = req.top_k or self._top_k
        # The cache key ignores route overrides and top_k, so bypass it when they are set.
        cacheable = use_cache and req.route is None and req.top_k is None

        embedding: Vector | None = None
        if cacheable and self._cache is not None:
            async with timer.stage("cache_exact"):
                hit = await self._cache.get_exact(query)
            if hit is None:
                async with timer.stage("embed"):
                    embedding = await self._safe_embed(query)
                if embedding is not None:
                    async with timer.stage("cache_semantic"):
                        hit = await self._cache.get_semantic(query, embedding)
            if hit is not None:
                cached = QueryResponse.model_validate(hit.response)
                resp = cached.model_copy(
                    update={
                        "query": query,
                        "cache": hit.kind,
                        "cache_similarity": round(hit.similarity, 4),
                        "timings_ms": {**timer.ms, "total": _ms(t0)},
                    }
                )
                self._observe(resp, t0)
                return resp

        if req.route is not None:
            decision = RouteDecision(Route(req.route), "override", "requested by caller")
        else:
            async with timer.stage("route"):
                decision = await self._router.route(query)

        if decision.route is not Route.SQL and embedding is None:
            async with timer.stage("embed"):
                embedding = await self._embedder.embed_query(query)

        resp = await self._dispatch(query, decision, embedding, timer, top_k=top_k)
        resp.cache = "miss" if cacheable else "bypass"
        resp.timings_ms = {**timer.ms, "total": _ms(t0)}
        if cacheable and self._cache is not None and resp.error is None:
            if embedding is None:
                embedding = await self._safe_embed(query)
            async with timer.stage("cache_write"):
                await self._cache.put(query, embedding, resp.model_dump(mode="json"))
        self._observe(resp, t0)
        return resp

    async def _dispatch(
        self,
        query: str,
        decision: RouteDecision,
        embedding: Vector | None,
        timer: _Timer,
        *,
        top_k: int,
    ) -> QueryResponse:
        route = RouteInfo(
            route=decision.route.value, source=decision.source, reason=decision.reason
        )
        base = QueryResponse(query=query, answer="", route=route)

        if decision.route is Route.DOCS:
            chunks = await self._retrieve_docs(query, embedding, timer, top_k=top_k)
            async with timer.stage("generate"):
                gen = await self._gen.answer_docs(query, chunks)
            return _with_answer(base, gen, chunks)

        if decision.route is Route.SQL:
            async with timer.stage("text2sql"):
                sql = await self._text2sql.run(query)
            base.sql = to_sql_payload(sql)
            if sql.result is None:
                base.answer = "I couldn't build a valid database query for that question."
                base.error = sql.errors[-1] if sql.errors else "text2sql failed"
                return base
            async with timer.stage("generate"):
                gen = await self._gen.answer_sql(query, sql.result)
            base.answer = gen.text
            return base

        # HYBRID: documents and SQL concurrently.
        async def _sql() -> Text2SQLResult:
            async with timer.stage("text2sql"):
                return await self._text2sql.run(query)

        chunks, sql = await asyncio.gather(
            self._retrieve_docs(query, embedding, timer, top_k=top_k), _sql()
        )
        base.sql = to_sql_payload(sql)
        async with timer.stage("generate"):
            gen = await self._gen.answer_hybrid(query, chunks, sql.result)
        return _with_answer(base, gen, chunks)

    async def _safe_embed(self, query: str) -> Vector | None:
        try:
            return await self._embedder.embed_query(query)
        except Exception as e:
            log.warning("query embedding failed (semantic cache skipped): %r", e)
            return None

    @staticmethod
    def _observe(resp: QueryResponse, t0: float) -> None:
        outcome: Literal["ok", "refused", "error"] = (
            "error" if resp.error else "refused" if resp.refused else "ok"
        )
        QUERIES.labels(resp.route.route, resp.cache, outcome).inc()
        QUERY_LATENCY.labels(resp.route.route, resp.cache).observe(time.perf_counter() - t0)


def _with_answer(
    base: QueryResponse, gen: GeneratedAnswer, chunks: Sequence[RetrievedChunk]
) -> QueryResponse:
    base.answer = gen.text
    base.refused = gen.refused
    base.refusal_reason = gen.refusal_reason
    base.sources = to_sources(chunks, gen.citations)
    return base


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
