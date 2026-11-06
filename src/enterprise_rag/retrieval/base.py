"""Retriever interface and the chunk value object every retriever returns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from enterprise_rag.ingestion.embedder import Vector


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: int  # surrogate PK (changes on re-ingest)
    document_id: str
    chunk_index: int
    title: str
    content: str
    score: float

    @property
    def key(self) -> str:
        """Stable citation key ``doc_id:chunk_index`` used in answers and eval labels."""
        return f"{self.document_id}:{self.chunk_index}"

    def with_score(self, score: float) -> RetrievedChunk:
        return replace(self, score=score)


class Retriever(ABC):
    name: str = "retriever"

    @abstractmethod
    async def retrieve(
        self, query: str, k: int, *, embedding: Vector | None = None
    ) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks, best first. ``embedding`` lets callers that already
        embedded the query (e.g. the semantic cache) avoid a second embedding call."""
