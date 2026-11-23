from __future__ import annotations

import fakeredis

from enterprise_rag.cache.response_cache import ResponseCache
from enterprise_rag.generation.answer import AnswerGenerator
from enterprise_rag.ingestion.embedder import HashEmbedder
from enterprise_rag.llm.base import Message
from enterprise_rag.retrieval.base import RetrievedChunk
from enterprise_rag.retrieval.reranker import NoopReranker
from enterprise_rag.routing.router import Route
from enterprise_rag.schemas import QueryRequest, RetrieveRequest
from enterprise_rag.service import RAGService
from enterprise_rag.sql.executor import SQLResult
from tests.fakes import FakeLLM, FakeRetriever, FakeRouter, FakeText2SQL

CHUNKS = [
    RetrievedChunk(i, f"doc{i}", 0, f"Doc {i}", f"policy text {i}", 1.0 / i) for i in range(1, 8)
]


def _reply(messages: list[Message]) -> str:
    system = messages[0].content
    if "database query results" in system:
        return "There are 42 trades."
    if "combining the database result" in system:
        return "42 trades; the limit is 5m [2]."
    return "The cap is 300 [1]."


def _service(
    route: Route = Route.DOCS, *, sql: SQLResult | None = None, cache: bool = False
) -> tuple[RAGService, FakeRouter, FakeRetriever, FakeText2SQL, FakeLLM]:
    router = FakeRouter(route)
    hybrid = FakeRetriever(CHUNKS)
    t2s = FakeText2SQL(sql if sql is not None else SQLResult(["n"], [[42]], 1, 1.0))
    llm = FakeLLM(_reply)
    svc = RAGService(
        embedder=HashEmbedder(),
        router=router,
        retrievers={"hybrid": hybrid, "vector": hybrid, "keyword": hybrid},
        reranker=NoopReranker(),
        text2sql=t2s,
        generator=AnswerGenerator(llm),
        cache=ResponseCache(fakeredis.FakeAsyncRedis(), None) if cache else None,
        candidates=20,
        top_k=5,
    )
    return svc, router, hybrid, t2s, llm


async def test_docs_route() -> None:
    svc, _, hybrid, t2s, _ = _service(Route.DOCS)
    r = await svc.answer(QueryRequest(query="hotel cap?"))
    assert r.route.route == "docs" and r.answer == "The cap is 300 [1]."
    assert len(r.sources) == 5 and [s.cited for s in r.sources] == [True] + [False] * 4
    assert hybrid.calls == [("hotel cap?", 20, True)]  # 20 candidates, shared embedding
    assert t2s.calls == 0 and r.sql is None
    assert {"route", "embed", "retrieve", "rerank", "generate", "total"} <= set(r.timings_ms)


async def test_sql_route_skips_retrieval() -> None:
    svc, _, hybrid, t2s, _ = _service(Route.SQL)
    r = await svc.answer(QueryRequest(query="How many trades?"))
    assert r.route.route == "sql" and r.answer == "There are 42 trades."
    assert r.sql is not None and r.sql.ok and r.sql.rows == [[42]]
    assert hybrid.calls == [] and t2s.calls == 1 and r.sources == []


async def test_sql_failure_reports_error() -> None:
    svc, *_ = _service(Route.SQL)
    svc._text2sql = FakeText2SQL(None)
    r = await svc.answer(QueryRequest(query="bad"))
    assert r.error == "relation does not exist" and r.sql is not None and not r.sql.ok


async def test_hybrid_uses_both() -> None:
    svc, _, hybrid, t2s, _ = _service(Route.HYBRID)
    r = await svc.answer(QueryRequest(query="trades vs limit"))
    assert r.route.route == "hybrid" and t2s.calls == 1 and len(hybrid.calls) == 1
    assert r.sources[1].cited and r.sql is not None and r.sql.ok


async def test_route_override_and_top_k() -> None:
    svc, router, _, t2s, _ = _service(Route.DOCS)
    r = await svc.answer(QueryRequest(query="anything", route="sql", top_k=3))
    assert router.calls == 0 and t2s.calls == 1
    assert r.route.source == "override" and r.cache == "bypass"


async def test_exact_cache_hit_on_repeat() -> None:
    svc, router, hybrid, _, llm = _service(Route.DOCS, cache=True)
    first = await svc.answer(QueryRequest(query="Hotel cap?"))
    second = await svc.answer(QueryRequest(query="hotel  cap"))
    assert first.cache == "miss" and second.cache == "exact"
    assert second.answer == first.answer and second.sources == first.sources
    assert router.calls == 1 and len(hybrid.calls) == 1 and len(llm.calls) == 1
    no_cache = await svc.answer(QueryRequest(query="hotel cap", use_cache=False))
    assert no_cache.cache == "bypass" and len(llm.calls) == 2


async def test_retrieve_endpoint_logic() -> None:
    svc, _, hybrid, _, _ = _service()
    r = await svc.retrieve(RetrieveRequest(query="q", mode="keyword", rerank=False, k=3))
    assert [s.n for s in r.sources] == [1, 2, 3]
    assert hybrid.calls == [("q", 3, False)]  # keyword: no embedding computed
