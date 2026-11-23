from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient

from enterprise_rag.api.app import create_app
from enterprise_rag.cache.response_cache import ResponseCache
from enterprise_rag.config import Settings
from enterprise_rag.retrieval.reranker import NoopReranker
from enterprise_rag.routing.router import Route
from tests.unit.test_service import _service


class FakeBackend:
    def __init__(self, status: str = "ok") -> None:
        self.settings = Settings()
        self.service, *_ = _service(Route.DOCS)
        self.cache = ResponseCache(fakeredis.FakeAsyncRedis(), None)
        self.reranker = NoopReranker()
        self.ingest: Any = None
        self.status = status
        self.closed = False

    async def health(self) -> dict[str, Any]:
        return {"status": self.status, "checks": {}}

    async def stats(self) -> dict[str, Any]:
        return {"documents_total": 3, "chunks": 9}

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(FakeBackend())) as c:  # type: ignore[arg-type]
        yield c


def test_query(client: TestClient) -> None:
    r = client.post("/query", json={"query": "hotel cap?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "The cap is 300 [1]." and body["route"]["route"] == "docs"
    assert body["sources"][0]["cited"] is True


def test_query_validation(client: TestClient) -> None:
    assert client.post("/query", json={"query": ""}).status_code == 422
    assert client.post("/query", json={"query": "x", "route": "web"}).status_code == 422


def test_retrieve(client: TestClient) -> None:
    r = client.post("/retrieve", json={"query": "q", "mode": "vector", "k": 2})
    assert r.status_code == 200 and len(r.json()["sources"]) == 2


def test_health_stats_cache_metrics(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/stats").json()["chunks"] == 9
    assert client.delete("/cache").json() == {"removed": 0}
    client.post("/query", json={"query": "metrics please"})
    m = client.get("/metrics")
    assert m.status_code == 200 and "rag_queries_total" in m.text


def test_health_down_is_503() -> None:
    with TestClient(create_app(FakeBackend("down"))) as c:  # type: ignore[arg-type]
        assert c.get("/health").status_code == 503


def test_injected_backend_is_not_closed_by_app() -> None:
    backend = FakeBackend()
    with TestClient(create_app(backend)):  # type: ignore[arg-type]
        pass
    assert not backend.closed
