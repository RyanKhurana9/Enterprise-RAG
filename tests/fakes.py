"""Test doubles shared by unit and integration tests."""

from __future__ import annotations

from collections.abc import Callable

from enterprise_rag.ingestion.embedder import Vector
from enterprise_rag.llm.base import LLM, Message
from enterprise_rag.retrieval.base import RetrievedChunk, Retriever
from enterprise_rag.routing.router import QueryRouter, Route, RouteDecision
from enterprise_rag.sql.executor import SQLResult
from enterprise_rag.sql.text2sql import Text2SQL, Text2SQLResult


class FakeLLM(LLM):
    """Returns scripted replies (a list consumed in order, or a function of the messages)."""

    def __init__(self, replies: list[str] | Callable[[list[Message]], str]) -> None:
        self._replies = replies
        self.calls: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], *, json_mode: bool = False, max_tokens: int | None = None
    ) -> str:
        self.calls.append(messages)
        if callable(self._replies):
            return self._replies(messages)
        if not self._replies:
            raise RuntimeError("FakeLLM exhausted")
        return self._replies.pop(0)


class FakeRetriever(Retriever):
    name = "fake"

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int, bool]] = []

    async def retrieve(
        self, query: str, k: int, *, embedding: Vector | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query, k, embedding is not None))
        return self.chunks[:k]


class FakeRouter(QueryRouter):
    def __init__(self, route: Route) -> None:
        self.route_to = route
        self.calls = 0

    async def route(self, query: str) -> RouteDecision:
        self.calls += 1
        return RouteDecision(self.route_to, "heuristic", "fake")


class FakeText2SQL(Text2SQL):
    """Returns a canned result without an LLM, schema or database."""

    def __init__(self, result: SQLResult | None, sql: str = "SELECT 1") -> None:
        self._result = result
        self._sql = sql
        self.calls = 0

    async def run(self, question: str) -> Text2SQLResult:
        self.calls += 1
        errors = [] if self._result else ["relation does not exist"]
        return Text2SQLResult(question, self._sql, self._result, 1, errors)
