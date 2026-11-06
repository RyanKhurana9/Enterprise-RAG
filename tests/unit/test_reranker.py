from __future__ import annotations

import os
from pathlib import Path

import pytest

from enterprise_rag.retrieval.base import RetrievedChunk
from enterprise_rag.retrieval.reranker import CrossEncoderReranker, NoopReranker


def chunk(i: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(i, f"doc{i}", 0, f"T{i}", content, score=1.0 / (i + 1))


class FakeCrossEncoder:
    """Scores a pair by how many query words the passage contains."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        self.calls += 1
        return [float(sum(w in p.split() for w in q.split())) for q, p in pairs]


async def test_cross_encoder_reorders_and_truncates() -> None:
    fake = FakeCrossEncoder()
    rr = CrossEncoderReranker(model=fake)
    chunks = [chunk(0, "nothing relevant"), chunk(1, "wire limit"), chunk(2, "wire")]
    out = await rr.rerank("wire limit", chunks, top_k=2)
    assert [c.chunk_id for c in out] == [1, 2]
    assert [c.score for c in out] == [2.0, 1.0]
    assert fake.calls == 1


async def test_cross_encoder_stable_on_ties_and_empty() -> None:
    rr = CrossEncoderReranker(model=FakeCrossEncoder())
    out = await rr.rerank("zzz", [chunk(0, "a"), chunk(1, "b")], top_k=5)
    assert [c.chunk_id for c in out] == [0, 1]  # equal scores keep retrieval order
    assert await rr.rerank("q", [], top_k=3) == []


async def test_noop_reranker() -> None:
    chunks = [chunk(i, "x") for i in range(4)]
    assert await NoopReranker().rerank("q", chunks, 2) == chunks[:2]


def _model_cached() -> bool:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return (hub / "models--cross-encoder--ms-marco-MiniLM-L-6-v2").exists()


@pytest.mark.skipif(not _model_cached(), reason="cross-encoder weights not cached locally")
async def test_real_cross_encoder_prefers_answering_passage() -> None:
    rr = CrossEncoderReranker(device="cpu")
    await rr.warmup()
    chunks = [
        chunk(0, "The cafeteria opens at 8am and serves breakfast until 10am."),
        chunk(1, "Outgoing wires above USD 250,000 require dual authorisation."),
    ]
    out = await rr.rerank("Which wire payments need two approvers?", chunks, top_k=2)
    assert out[0].chunk_id == 1
    assert out[0].score > out[1].score
