"""FastAPI application.

    uvicorn enterprise_rag.api.app:create_app --factory --port 8000

Endpoints:
    POST   /query     answer a question (routing, retrieval, Text2SQL, generation, cache)
    POST   /retrieve  inspect retrieval only (vector | keyword | hybrid, +/- rerank)
    POST   /ingest    (re)ingest the document directory (idempotent)
    GET    /health    dependency probes (postgres, redis, ollama)
    GET    /stats     corpus / cache statistics and active configuration
    DELETE /cache     clear the exact and semantic caches
    GET    /metrics   Prometheus metrics
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from enterprise_rag.cache.response_cache import ResponseCache
from enterprise_rag.config import Settings, get_settings
from enterprise_rag.ingestion.pipeline import IngestPipeline
from enterprise_rag.retrieval.reranker import CrossEncoderReranker, Reranker
from enterprise_rag.schemas import QueryRequest, QueryResponse, RetrieveRequest, RetrieveResponse
from enterprise_rag.service import RAGService

log = logging.getLogger(__name__)


class Backend(Protocol):
    """What the API needs from the composition root (``Container`` satisfies it)."""

    settings: Settings
    service: RAGService
    cache: ResponseCache
    ingest: IngestPipeline
    reranker: Reranker

    async def health(self) -> dict[str, Any]: ...
    async def stats(self) -> dict[str, Any]: ...
    async def aclose(self) -> None: ...


def _backend(request: Request) -> Backend:
    backend: Backend = request.app.state.backend
    return backend


def create_app(backend: Backend | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = backend is None
        if backend is None:
            from enterprise_rag.container import Container

            app.state.backend = Container.build(get_settings())
        else:
            app.state.backend = backend
        b: Backend = app.state.backend
        warmup: asyncio.Task[None] | None = None
        if isinstance(b.reranker, CrossEncoderReranker):
            # Load the cross-encoder in the background so /health is up immediately.
            warmup = asyncio.create_task(b.reranker.warmup())
        try:
            yield
        finally:
            if warmup is not None and not warmup.done():
                warmup.cancel()
            if owned:
                await b.aclose()

    app = FastAPI(
        title="Enterprise RAG",
        version="0.1.0",
        description="Hybrid RAG over policies, 10-Ks and FAQs plus Text2SQL over a trading "
        "warehouse.",
        lifespan=lifespan,
    )

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest, request: Request) -> QueryResponse:
        try:
            return await _backend(request).service.answer(req)
        except Exception as e:
            log.exception("query failed")
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(req: RetrieveRequest, request: Request) -> RetrieveResponse:
        try:
            return await _backend(request).service.retrieve(req)
        except Exception as e:
            log.exception("retrieve failed")
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e

    @app.post("/ingest")
    async def ingest(request: Request) -> dict[str, Any]:
        b = _backend(request)
        report = await b.ingest.ingest_directory(Path(b.settings.docs_dir))
        return {
            "docs_seen": report.docs_seen,
            "docs_ingested": report.docs_ingested,
            "docs_skipped": report.docs_skipped,
            "chunks_written": report.chunks_written,
            "embeddings_computed": report.embeddings_computed,
            "embeddings_reused": report.embeddings_reused,
            "seconds": round(report.seconds, 2),
            "errors": report.errors,
        }

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        result = await _backend(request).health()
        return JSONResponse(result, status_code=503 if result["status"] == "down" else 200)

    @app.get("/stats")
    async def stats(request: Request) -> dict[str, Any]:
        return await _backend(request).stats()

    @app.delete("/cache")
    async def clear_cache(request: Request) -> dict[str, int]:
        return {"removed": await _backend(request).cache.clear()}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
