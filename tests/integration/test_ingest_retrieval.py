"""Stage 2: ingestion pipeline + vector/keyword/hybrid retrieval against real Postgres."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text

from enterprise_rag.config import Settings
from enterprise_rag.db.engine import SessionFactory, create_engine, create_session_factory
from enterprise_rag.ingestion.chunker import RecursiveChunker, WhitespaceTokenizer
from enterprise_rag.ingestion.embedder import HashEmbedder
from enterprise_rag.ingestion.loaders import default_loader
from enterprise_rag.ingestion.pipeline import IngestPipeline
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.sql_retrievers import KeywordRetriever, VectorRetriever

pytestmark = pytest.mark.integration

DOCS = {
    "policies/wire.md": "# Wire Policy\n\nOutgoing wires above USD 250,000 need dual approval "
    "by two treasury officers.\n\nCall-back verification is mandatory.",
    "policies/travel.md": "# Travel Policy\n\nBusiness class is allowed on flights longer "
    "than six hours.\n\nHotel caps apply in London.",
    "faq/payroll.md": "# Payroll FAQ\n\nSalary is paid by direct deposit on the 25th.",
}


@pytest.fixture(scope="module")
async def sf(pg_url: str) -> AsyncIterator[SessionFactory]:
    engine = create_engine(pg_url, Settings(), application_name="test")
    yield create_session_factory(engine)
    await engine.dispose()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("docs")
    for rel, body in DOCS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body)
    return root


def _pipeline(sf: SessionFactory, chunk_size: int = 64) -> IngestPipeline:
    chunker = RecursiveChunker(WhitespaceTokenizer(), chunk_size=chunk_size, overlap=4)
    return IngestPipeline(sf, default_loader(), chunker, HashEmbedder())


async def test_ingest_is_idempotent(sf: SessionFactory, corpus: Path) -> None:
    first = await _pipeline(sf).ingest_directory(corpus)
    assert first.docs_ingested == 3 and first.chunks_written >= 3 and not first.errors
    second = await _pipeline(sf).ingest_directory(corpus)
    assert second.docs_ingested == 0 and second.docs_skipped == 3
    async with sf() as s:
        assert (await s.execute(text("SELECT count(*) FROM documents"))).scalar_one() == 3


async def test_changed_doc_is_replaced_and_unchanged_chunks_reuse_embeddings(
    sf: SessionFactory, corpus: Path
) -> None:
    await _pipeline(sf).ingest_directory(corpus)
    p = corpus / "faq/payroll.md"
    original = p.read_text()
    p.write_text(original + "\n\nBonuses are paid in March.")
    try:
        r = await _pipeline(sf, chunk_size=12).ingest_paths([p])
        assert r.docs_ingested == 1
        async with sf() as s:
            n = (
                await s.execute(text("SELECT count(*) FROM chunks WHERE document_id = 'payroll'"))
            ).scalar_one()
        assert n == r.chunks_written
    finally:
        p.write_text(original)


async def test_retrievers(sf: SessionFactory, corpus: Path) -> None:
    await _pipeline(sf).ingest_directory(corpus)
    emb = HashEmbedder()
    vector = VectorRetriever(sf, emb)
    keyword = KeywordRetriever(sf)
    hybrid = HybridRetriever([vector, keyword], embedder=emb, candidates_per_retriever=5)

    q = "which wires need two treasury officers"
    for r in (vector, keyword, hybrid):
        hits = await r.retrieve(q, 2)
        assert hits, r.name
        assert hits[0].document_id == "wire", r.name
        assert hits[0].key.startswith("wire:")

    kw = await keyword.retrieve("flights hotel London", 3)
    assert kw[0].document_id == "travel"
    assert await keyword.retrieve("the and of", 3) == []  # only stop-words -> no tsquery
    fused = await hybrid.retrieve(q, 3)
    assert fused[0].score == pytest.approx(2 / 61)  # rank 1 in both lists
