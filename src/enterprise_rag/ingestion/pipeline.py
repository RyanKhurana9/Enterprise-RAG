"""Idempotent ingestion: load -> chunk -> embed -> batch insert.

Idempotency is content-addressed:

* a document whose sha256 matches the stored ``documents.content_hash`` is skipped entirely
  (no chunking, no embedding, no writes);
* a changed document has its chunks replaced in one transaction, and chunks whose own hash
  already exists anywhere in the table reuse the stored embedding instead of calling the
  embedding model again (embedding is by far the most expensive step).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import distinct_on, insert

from enterprise_rag.db.engine import SessionFactory
from enterprise_rag.db.models import Chunk, Document
from enterprise_rag.ingestion.chunker import RecursiveChunker, TextChunk
from enterprise_rag.ingestion.embedder import Embedder, Vector
from enterprise_rag.ingestion.loaders import CompositeLoader, LoadedDocument

log = logging.getLogger(__name__)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class IngestReport:
    docs_seen: int = 0
    docs_ingested: int = 0
    docs_skipped: int = 0
    chunks_written: int = 0
    embeddings_computed: int = 0
    embeddings_reused: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class _Pending:
    doc: LoadedDocument
    doc_hash: str
    chunks: list[TextChunk]


class IngestPipeline:
    def __init__(
        self,
        session_factory: SessionFactory,
        loader: CompositeLoader,
        chunker: RecursiveChunker,
        embedder: Embedder,
        *,
        docs_per_batch: int = 32,
    ) -> None:
        self._sf = session_factory
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._docs_per_batch = docs_per_batch
        # Part of each document's stored hash: changing the chunking strategy invalidates
        # every document, so a re-run re-chunks instead of silently keeping stale chunks.
        self.version = (
            f"chunker=v2;size={chunker.chunk_size};overlap={chunker.overlap};header=title"
        )

    async def ingest_directory(self, root: Path) -> IngestReport:
        return await self.ingest_paths(self._loader.discover(root))

    async def ingest_paths(self, paths: Iterable[Path]) -> IngestReport:
        t0 = time.perf_counter()
        report = IngestReport()
        docs: list[LoadedDocument] = []
        for path in paths:
            report.docs_seen += 1
            try:
                docs.append(self._loader.load(path))
            except Exception as exc:  # one bad file must not abort the batch
                log.warning("failed to load %s: %s", path, exc)
                report.errors.append(f"{path}: {exc}")

        existing = await self._existing_hashes([d.doc_id for d in docs])
        pending: list[_Pending] = []
        for doc in docs:
            h = sha256(f"{self.version}\n{doc.text}")
            if existing.get(doc.doc_id) == h:
                report.docs_skipped += 1
                continue
            header = f"Document: {doc.title}\n\n"
            pending.append(_Pending(doc, h, self._chunker.chunk(doc.text, header=header)))

        for i in range(0, len(pending), self._docs_per_batch):
            await self._write_batch(pending[i : i + self._docs_per_batch], report)

        report.seconds = time.perf_counter() - t0
        return report

    async def _existing_hashes(self, doc_ids: list[str]) -> dict[str, str]:
        if not doc_ids:
            return {}
        async with self._sf() as s:
            rows = await s.execute(
                select(Document.id, Document.content_hash).where(Document.id.in_(doc_ids))
            )
            return {r[0]: r[1] for r in rows}

    async def _embeddings_for(
        self, texts: list[str], hashes: list[str], report: IngestReport
    ) -> list[Vector]:
        async with self._sf() as s:
            rows = await s.execute(
                select(Chunk.content_hash, Chunk.embedding)
                .where(Chunk.content_hash.in_(set(hashes)))
                .ext(distinct_on(Chunk.content_hash))
            )
            known: dict[str, Vector] = {r[0]: np.asarray(r[1], np.float32) for r in rows}
        missing = [i for i, h in enumerate(hashes) if h not in known]
        fresh = await self._embedder.embed_documents([texts[i] for i in missing])
        for i, vec in zip(missing, fresh, strict=True):
            known[hashes[i]] = vec
        report.embeddings_computed += len(missing)
        report.embeddings_reused += len(hashes) - len(missing)
        return [known[h] for h in hashes]

    async def _write_batch(self, batch: list[_Pending], report: IngestReport) -> None:
        texts = [c.text for p in batch for c in p.chunks]
        hashes = [sha256(t) for t in texts]
        vectors = await self._embeddings_for(texts, hashes, report)

        rows = []
        k = 0
        for p in batch:
            for c in p.chunks:
                rows.append(
                    {
                        "document_id": p.doc.doc_id,
                        "chunk_index": c.index,
                        "content": c.text,
                        "token_count": c.token_count,
                        "content_hash": hashes[k],
                        "embedding": vectors[k],
                    }
                )
                k += 1

        async with self._sf() as s, s.begin():
            doc_rows = [
                {
                    "id": p.doc.doc_id,
                    "source_path": p.doc.source_path,
                    "title": p.doc.title,
                    "doc_type": p.doc.doc_type,
                    "content_hash": p.doc_hash,
                }
                for p in batch
            ]
            stmt = insert(Document).values(doc_rows)
            await s.execute(
                stmt.on_conflict_do_update(
                    index_elements=[Document.id],
                    set_={
                        "source_path": stmt.excluded.source_path,
                        "title": stmt.excluded.title,
                        "doc_type": stmt.excluded.doc_type,
                        "content_hash": stmt.excluded.content_hash,
                        "updated_at": stmt.excluded.updated_at,
                    },
                )
            )
            await s.execute(
                delete(Chunk).where(Chunk.document_id.in_([p.doc.doc_id for p in batch]))
            )
            if rows:
                # executemany through asyncpg's prepared-statement batching
                await s.execute(insert(Chunk), rows)
        report.docs_ingested += len(batch)
        report.chunks_written += len(rows)
