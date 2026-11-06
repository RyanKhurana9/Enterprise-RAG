"""Second-stage reranking.

Bi-encoder (vector) and BM25-style retrieval score query and passage *independently*; a
cross-encoder reads them *together* and is far more precise, but costs one transformer
forward pass per (query, passage) pair. So we only apply it to the top-N fused candidates
(default 20 -> 5): precision where it matters, bounded latency.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from enterprise_rag.retrieval.base import RetrievedChunk

log = logging.getLogger(__name__)


class Reranker(ABC):
    name: str = "reranker"

    @abstractmethod
    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the best ``top_k`` chunks, re-scored, best first."""


class NoopReranker(Reranker):
    name = "noop"

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return list(chunks[:top_k])


class CrossEncoderReranker(Reranker):
    """sentence-transformers CrossEncoder (default ms-marco-MiniLM-L-6-v2, 22M params).

    * The model is loaded lazily (first call or ``warmup()``), so importing the API is cheap.
    * Inference is CPU/GPU-bound, so it runs in a worker thread to keep the event loop free;
      a semaphore serialises calls because concurrent forward passes on one model just
      contend for the same cores.
    * Scores are raw logits (identity activation) – comparable across queries, which lets
      the generator refuse when even the best passage scores below a threshold.
    """

    name = "cross-encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = model  # injectable for tests
        self._lock = asyncio.Lock()

    def _load(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            log.info("loading cross-encoder %s", self.model_name)
            self._model = CrossEncoder(
                self.model_name,
                device=self._device,
                max_length=self._max_length,
                activation_fn=torch.nn.Identity(),
            )
        return self._model

    async def warmup(self) -> None:
        await self.rerank("warmup", [RetrievedChunk(0, "w", 0, "w", "warmup", 0.0)], 1)

    def _score(self, query: str, passages: list[str]) -> list[float]:
        model = self._load()
        scores = model.predict(
            [(query, p) for p in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        async with self._lock:
            scores = await asyncio.to_thread(self._score, query, [c.content for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
        return [chunks[i].with_score(scores[i]) for i in order[:top_k]]
