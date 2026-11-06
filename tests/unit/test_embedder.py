from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from enterprise_rag.ingestion.embedder import HashEmbedder, OllamaEmbedder
from enterprise_rag.llm.ollama import OllamaError


def _client(calls: list[dict[str, object]], dim: int = 4, status: int = 200) -> httpx.AsyncClient:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        calls.append(body)
        n = len(body["input"])
        return httpx.Response(status, json={"embeddings": [[3.0, 4.0] + [0.0] * (dim - 2)] * n})

    return httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))


async def test_ollama_embedder_batches_prefixes_and_normalises() -> None:
    calls: list[dict[str, object]] = []
    emb = OllamaEmbedder(_client(calls), dim=4, batch_size=2)
    vecs = await emb.embed_documents(["a", "b", "c"])
    assert len(calls) == 2  # 2 + 1
    assert calls[0]["input"] == ["search_document: a", "search_document: b"]
    assert np.allclose(vecs[0], [0.6, 0.8, 0, 0])


async def test_query_embedding_is_cached() -> None:
    calls: list[dict[str, object]] = []
    emb = OllamaEmbedder(_client(calls), dim=4, query_cache_size=1)
    await emb.embed_query("q1")
    await emb.embed_query(" q1 ")
    assert len(calls) == 1 and calls[0]["input"] == ["search_query: q1"]
    await emb.embed_query("q2")  # evicts q1
    await emb.embed_query("q1")
    assert len(calls) == 3


async def test_dimension_mismatch_and_http_error() -> None:
    with pytest.raises(ValueError):
        await OllamaEmbedder(_client([], dim=4), dim=8).embed_query("x")
    with pytest.raises(OllamaError):
        await OllamaEmbedder(_client([], status=500), dim=4).embed_query("x")


async def test_hash_embedder_similarity() -> None:
    e = HashEmbedder(dim=256)
    a, b, c = await e.embed_documents(["wire transfer limit", "wire transfer cap", "pension"])
    assert float(a @ b) > float(a @ c)
    assert np.isclose(np.linalg.norm(await e.embed_query("x y")), 1.0)
    assert not (await e.embed_query("...")).any()
