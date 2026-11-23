"""Composition root: the only place that knows concrete classes.

Every component receives its collaborators through its constructor, so tests (and
alternative deployments) can swap any piece – e.g. a fake LLM or a HashEmbedder.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from enterprise_rag.cache.response_cache import ResponseCache
from enterprise_rag.config import Settings
from enterprise_rag.db.engine import SessionFactory, create_engine, create_session_factory
from enterprise_rag.generation.answer import AnswerGenerator
from enterprise_rag.ingestion.chunker import RecursiveChunker, TiktokenTokenizer
from enterprise_rag.ingestion.embedder import Embedder, OllamaEmbedder
from enterprise_rag.ingestion.loaders import default_loader
from enterprise_rag.ingestion.pipeline import IngestPipeline
from enterprise_rag.llm.base import LLM
from enterprise_rag.llm.ollama import OllamaLLM
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import CrossEncoderReranker, NoopReranker, Reranker
from enterprise_rag.retrieval.sql_retrievers import KeywordRetriever, VectorRetriever
from enterprise_rag.routing.router import (
    CascadeRouter,
    HeuristicRouter,
    LLMRouter,
    QueryRouter,
)
from enterprise_rag.service import RAGService
from enterprise_rag.sql.executor import SQLExecutor
from enterprise_rag.sql.guard import SQLGuard
from enterprise_rag.sql.introspector import SchemaIntrospector
from enterprise_rag.sql.text2sql import Text2SQL

SCHEMA_NOTES = (
    "-- trades.traded_at spans 2023-01-01 to 2025-06-30 (UTC).",
    "-- trades.price is per unit; notional = qty * price.",
)


@dataclass
class Container:
    settings: Settings
    engine: AsyncEngine
    session_factory: SessionFactory
    http: httpx.AsyncClient
    embedder: Embedder
    vector: VectorRetriever
    keyword: KeywordRetriever
    hybrid: HybridRetriever
    reranker: Reranker
    ingest: IngestPipeline
    readonly_engine: AsyncEngine
    llm: LLM
    introspector: SchemaIntrospector
    guard: SQLGuard
    text2sql: Text2SQL
    heuristic_router: HeuristicRouter
    llm_router: LLMRouter
    router: QueryRouter
    redis: Redis
    cache: ResponseCache
    generator: AnswerGenerator
    service: RAGService

    @classmethod
    def build(cls, settings: Settings) -> Container:
        engine = create_engine(settings.database_url, settings, application_name="rag-api")
        sf = create_session_factory(engine)
        http = httpx.AsyncClient(
            base_url=settings.ollama_url,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )
        embedder = OllamaEmbedder(
            http,
            settings.embed_model,
            dim=settings.embed_dim,
            batch_size=settings.embed_batch_size,
        )
        vector = VectorRetriever(sf, embedder, ef_search=settings.hnsw_ef_search)
        keyword = KeywordRetriever(sf)
        hybrid = HybridRetriever(
            [vector, keyword],
            embedder=embedder,
            rrf_k=settings.rrf_k,
            candidates_per_retriever=settings.retrieval_candidates,
        )
        chunker = RecursiveChunker(
            TiktokenTokenizer(), settings.chunk_size_tokens, settings.chunk_overlap_tokens
        )
        reranker: Reranker = (
            CrossEncoderReranker(settings.rerank_model, device=settings.rerank_device)
            if settings.rerank_enabled
            else NoopReranker()
        )
        ingest = IngestPipeline(sf, default_loader(), chunker, embedder)

        # Text2SQL gets its own pool, authenticated as the read-only role.
        readonly = create_engine(
            settings.readonly_database_url, settings, application_name="rag-sql", vector=False
        )
        llm = OllamaLLM(
            http,
            settings.llm_model,
            temperature=settings.llm_temperature,
            num_ctx=settings.llm_num_ctx,
        )
        introspector = SchemaIntrospector(readonly, settings.sql_allowed_tables, notes=SCHEMA_NOTES)
        guard = SQLGuard(settings.sql_allowed_tables, max_rows=settings.sql_max_rows)
        executor = SQLExecutor(readonly, statement_timeout_ms=settings.sql_statement_timeout_ms)
        text2sql = Text2SQL(
            llm, introspector, guard, executor, repair_attempts=settings.sql_repair_attempts
        )
        heuristic = HeuristicRouter()
        llm_router = LLMRouter(llm, heuristic)
        router: QueryRouter = {
            "llm": llm_router,
            "heuristic": heuristic,
            "cascade": CascadeRouter(
                heuristic, llm_router, min_margin=settings.router_cascade_margin
            ),
        }[settings.router_mode]

        redis = Redis.from_url(settings.redis_url, socket_timeout=1.0, socket_connect_timeout=1.0)
        cache = ResponseCache(
            redis if settings.cache_enabled else None,
            sf if settings.cache_enabled else None,
            exact_ttl_s=settings.exact_cache_ttl_s,
            semantic_threshold=settings.semantic_cache_threshold,
            semantic_ttl_s=settings.semantic_cache_ttl_s,
            max_entries=settings.semantic_cache_max_entries,
        )
        # Pre-LLM refusal only makes sense on cross-encoder logits, not RRF scores.
        generator = AnswerGenerator(
            llm, min_score=settings.min_rerank_score if settings.rerank_enabled else None
        )
        service = RAGService(
            embedder=embedder,
            router=router,
            retrievers={"vector": vector, "keyword": keyword, "hybrid": hybrid},
            reranker=reranker,
            text2sql=text2sql,
            generator=generator,
            cache=cache if settings.cache_enabled else None,
            candidates=settings.retrieval_candidates,
            top_k=settings.final_top_k,
        )
        return cls(
            settings=settings,
            engine=engine,
            session_factory=sf,
            http=http,
            embedder=embedder,
            vector=vector,
            keyword=keyword,
            hybrid=hybrid,
            reranker=reranker,
            ingest=ingest,
            readonly_engine=readonly,
            llm=llm,
            introspector=introspector,
            guard=guard,
            text2sql=text2sql,
            heuristic_router=heuristic,
            llm_router=llm_router,
            router=router,
            redis=redis,
            cache=cache,
            generator=generator,
            service=service,
        )

    async def health(self) -> dict[str, Any]:
        """Probe every dependency concurrently; each probe is bounded to 3s."""

        async def postgres() -> str:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return "ok"

        async def redis() -> str:
            await self.redis.ping()
            return "ok"

        async def ollama() -> str:
            resp = await self.http.get("/api/tags", timeout=3.0)
            resp.raise_for_status()
            names = {m["name"] for m in resp.json().get("models", [])}
            missing = [
                m
                for m in (self.settings.llm_model, self.settings.embed_model)
                if m not in names and f"{m}:latest" not in names
            ]
            return "ok" if not missing else f"missing models: {', '.join(missing)}"

        async def probe(fn: Any) -> dict[str, Any]:
            t0 = time.perf_counter()
            try:
                status = await asyncio.wait_for(fn(), timeout=3.0)
            except Exception as e:
                status = f"error: {type(e).__name__}: {str(e)[:120]}"
            return {"status": status, "ms": round((time.perf_counter() - t0) * 1000, 1)}

        names = ("postgres", "redis", "ollama")
        results = await asyncio.gather(*(probe(f) for f in (postgres, redis, ollama)))
        checks = dict(zip(names, results, strict=True))
        # Redis is only a cache: the API still answers without it.
        critical_ok = all(checks[n]["status"] == "ok" for n in ("postgres", "ollama"))
        degraded = checks["redis"]["status"] != "ok"
        status = "ok" if critical_ok and not degraded else "degraded" if critical_ok else "down"
        return {"status": status, "checks": checks}

    async def stats(self) -> dict[str, Any]:
        async with self.session_factory() as s:
            docs = (
                await s.execute(
                    text("SELECT doc_type, count(*) FROM documents GROUP BY doc_type ORDER BY 1")
                )
            ).all()
            chunks = (await s.execute(text("SELECT count(*) FROM chunks"))).scalar_one()
        return {
            "documents": {t: int(n) for t, n in docs},
            "documents_total": sum(int(n) for _, n in docs),
            "chunks": int(chunks),
            "cache": await self.cache.stats(),
            "config": {
                "llm_model": self.settings.llm_model,
                "embed_model": self.settings.embed_model,
                "router_mode": self.settings.router_mode,
                "rerank_enabled": self.settings.rerank_enabled,
                "rerank_model": self.settings.rerank_model,
                "retrieval_candidates": self.settings.retrieval_candidates,
                "final_top_k": self.settings.final_top_k,
                "semantic_cache_threshold": self.settings.semantic_cache_threshold,
                "cache_enabled": self.settings.cache_enabled,
            },
        }

    async def aclose(self) -> None:
        await self.redis.aclose()
        await self.http.aclose()
        await self.engine.dispose()
        await self.readonly_engine.dispose()
