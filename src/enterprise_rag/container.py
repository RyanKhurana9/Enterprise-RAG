"""Composition root: the only place that knows concrete classes.

Every component receives its collaborators through its constructor, so tests (and
alternative deployments) can swap any piece – e.g. a fake LLM or a HashEmbedder.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine

from enterprise_rag.config import Settings
from enterprise_rag.db.engine import SessionFactory, create_engine, create_session_factory
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
        )

    async def aclose(self) -> None:
        await self.http.aclose()
        await self.engine.dispose()
        await self.readonly_engine.dispose()
