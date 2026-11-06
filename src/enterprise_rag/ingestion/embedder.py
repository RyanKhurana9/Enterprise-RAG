"""Embedders. Documents and queries are embedded differently (nomic task prefixes)."""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict

import httpx
import numpy as np
import numpy.typing as npt

from enterprise_rag.llm.ollama import ollama_embed

Vector = npt.NDArray[np.float32]


class Embedder(ABC):
    dim: int

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> Vector: ...


class OllamaEmbedder(Embedder):
    """nomic-embed-text via Ollama.

    nomic models are trained with task prefixes; using ``search_document:`` /
    ``search_query:`` measurably improves retrieval versus raw text. Vectors are
    L2-normalised so cosine distance == 1 - dot product. A small LRU avoids re-embedding
    repeated queries (the semantic cache and the vector retriever need the same vector).
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str = "nomic-embed-text",
        *,
        dim: int = 768,
        batch_size: int = 32,
        max_concurrency: int = 2,
        query_cache_size: int = 2048,
    ) -> None:
        self._client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._sem = asyncio.Semaphore(max_concurrency)
        self._qcache: OrderedDict[str, Vector] = OrderedDict()
        self._qcache_size = query_cache_size

    async def _embed(self, texts: list[str]) -> list[Vector]:
        async with self._sem:
            raw = await ollama_embed(self._client, self.model, texts)
        out = []
        for v in raw:
            arr = np.asarray(v, dtype=np.float32)
            if arr.shape != (self.dim,):
                raise ValueError(f"expected dim {self.dim}, got {arr.shape}")
            out.append(arr / (np.linalg.norm(arr) or 1.0))
        return out

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        batches = [
            [f"search_document: {t}" for t in texts[i : i + self.batch_size]]
            for i in range(0, len(texts), self.batch_size)
        ]
        results = await asyncio.gather(*(self._embed(b) for b in batches))
        return [v for batch in results for v in batch]

    async def embed_query(self, text: str) -> Vector:
        key = text.strip()
        if (hit := self._qcache.get(key)) is not None:
            self._qcache.move_to_end(key)
            return hit
        (vec,) = await self._embed([f"search_query: {key}"])
        self._qcache[key] = vec
        if len(self._qcache) > self._qcache_size:
            self._qcache.popitem(last=False)
        return vec


class HashEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder: no model, no network. For tests.

    Texts sharing words get similar vectors, which is enough to exercise the retrieval and
    cache code paths end to end.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def _vec(self, text: str) -> Vector:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            w = word.strip(".,;:?!()[]\"'")
            if w:
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "big")
                v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> Vector:
        return self._vec(text)
