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
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.sql_retrievers import KeywordRetriever, VectorRetriever


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
    ingest: IngestPipeline

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
        ingest = IngestPipeline(sf, default_loader(), chunker, embedder)
        return cls(settings, engine, sf, http, embedder, vector, keyword, hybrid, ingest)

    async def aclose(self) -> None:
        await self.http.aclose()
        await self.engine.dispose()
