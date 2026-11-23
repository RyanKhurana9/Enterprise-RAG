"""Request/response models shared by the service layer, the API and the cache."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RouteName = Literal["docs", "sql", "hybrid"]
CacheStatus = Literal["miss", "exact", "semantic", "bypass"]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    route: RouteName | None = Field(
        default=None, description="Force a route; omit to let the router decide."
    )
    use_cache: bool = True
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    n: int  # the [n] number shown to the LLM and used in the answer text
    key: str  # stable doc_id:chunk_index
    document_id: str
    chunk_index: int
    title: str
    score: float
    snippet: str
    cited: bool = False


class SQLPayload(BaseModel):
    sql: str | None
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    attempts: int = 0
    errors: list[str] = []
    ok: bool = False


class RouteInfo(BaseModel):
    route: RouteName
    source: str  # llm | heuristic | override | cache
    reason: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    route: RouteInfo
    sources: list[Source] = []
    sql: SQLPayload | None = None
    refused: bool = False
    refusal_reason: str = ""
    cache: CacheStatus = "miss"
    cache_similarity: float | None = None
    timings_ms: dict[str, float] = {}
    error: str | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    rerank: bool = True
    k: int = Field(default=5, ge=1, le=50)


class RetrieveResponse(BaseModel):
    query: str
    mode: str
    reranked: bool
    sources: list[Source]
    timings_ms: dict[str, float] = {}
