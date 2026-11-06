"""Postgres-backed retrievers: pgvector HNSW (cosine) and tsvector full-text."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from enterprise_rag.db.engine import SessionFactory
from enterprise_rag.ingestion.embedder import Embedder, Vector
from enterprise_rag.retrieval.base import RetrievedChunk, Retriever

# The inner query touches only `chunks` and orders by the distance operator, which is the
# exact shape the HNSW index can serve (ORDER BY embedding <=> $q LIMIT k). Joining to
# `documents` happens after the LIMIT, on at most k rows.
VECTOR_SQL = text(
    """
    WITH nn AS (
        SELECT id, document_id, chunk_index, content, embedding <=> :q AS distance
        FROM chunks
        ORDER BY embedding <=> :q
        LIMIT :k
    )
    SELECT nn.id, nn.document_id, nn.chunk_index, d.title, nn.content,
           1 - nn.distance AS score
    FROM nn JOIN documents d ON d.id = nn.document_id
    ORDER BY nn.distance
    """
)

# Natural-language questions rarely contain *all* their words in one chunk, so an AND query
# (plainto_tsquery / websearch_to_tsquery) has poor recall. We take the query's lexemes
# (stemmed, stop-words removed by to_tsvector) and OR them; ts_rank_cd then rewards chunks
# that contain more of them, close together. Normalisation flag 1 divides by 1 + log(length)
# – a BM25-like document-length penalty. `tsv @@ tsq` is answered by the GIN index.
KEYWORD_SQL = text(
    """
    WITH q AS (
        SELECT to_tsquery('english', string_agg(quote_literal(lexeme), ' | ')) AS tsq
        FROM unnest(to_tsvector('english', :query))
    ), hits AS (
        SELECT c.id, c.document_id, c.chunk_index, c.content,
               ts_rank_cd(c.tsv, q.tsq, 1) AS score
        FROM chunks c, q
        WHERE c.tsv @@ q.tsq
        ORDER BY score DESC, c.id
        LIMIT :k
    )
    SELECT h.id, h.document_id, h.chunk_index, d.title, h.content, h.score
    FROM hits h JOIN documents d ON d.id = h.document_id
    ORDER BY h.score DESC, h.id
    """
)


def _rows_to_chunks(rows: Any) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=r.id,
            document_id=r.document_id,
            chunk_index=r.chunk_index,
            title=r.title,
            content=r.content,
            score=float(r.score),
        )
        for r in rows
    ]


class VectorRetriever(Retriever):
    name = "vector"

    def __init__(
        self, session_factory: SessionFactory, embedder: Embedder, *, ef_search: int = 64
    ) -> None:
        self._sf = session_factory
        self._embedder = embedder
        self._ef_search = int(ef_search)

    async def retrieve(
        self, query: str, k: int, *, embedding: Vector | None = None
    ) -> list[RetrievedChunk]:
        vec = embedding if embedding is not None else await self._embedder.embed_query(query)
        async with self._sf() as s, s.begin():
            # ef_search >= k or HNSW can return fewer than k rows. SET LOCAL scopes it to
            # this transaction so pooled connections are not left modified.
            await s.execute(text(f"SET LOCAL hnsw.ef_search = {max(self._ef_search, k)}"))
            rows = await s.execute(VECTOR_SQL, {"q": vec, "k": k})
            return _rows_to_chunks(rows)


class KeywordRetriever(Retriever):
    name = "keyword"

    def __init__(self, session_factory: SessionFactory) -> None:
        self._sf = session_factory

    async def retrieve(
        self, query: str, k: int, *, embedding: Vector | None = None
    ) -> list[RetrievedChunk]:
        async with self._sf() as s:
            rows = await s.execute(KEYWORD_SQL, {"query": query, "k": k})
            return _rows_to_chunks(rows)
