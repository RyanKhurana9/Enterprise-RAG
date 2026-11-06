from __future__ import annotations

import asyncio

import pytest

from enterprise_rag.llm.base import LLM, Message
from enterprise_rag.routing.router import (
    CascadeRouter,
    HeuristicRouter,
    LLMRouter,
    Route,
)
from tests.fakes import FakeLLM

H = HeuristicRouter()


@pytest.mark.parametrize(
    ("q", "route"),
    [
        ("How many trades did EMEA clients make in 2024?", Route.SQL),
        ("What is the total notional traded in AAPL?", Route.SQL),
        ("Which 5 tickers have the most trades?", Route.SQL),
        ("What is the hotel cap in the Wealth Management travel policy?", Route.DOCS),
        ("Where is Becker Financial headquartered?", Route.DOCS),
        ("How long must staff hold a stock before selling it?", Route.DOCS),
        ("Tell me something", Route.DOCS),  # no signal -> docs (safe default)
    ],
)
async def test_heuristic(q: str, route: Route) -> None:
    d = await H.route(q)
    assert d.route is route and d.source == "heuristic"


async def test_heuristic_hybrid_when_both_strong_and_close() -> None:
    d = await H.route("Do any trades in AMER accounts breach the Global Markets policy limit?")
    assert d.route is Route.HYBRID


async def test_llm_router_parses_reply() -> None:
    llm = FakeLLM(['{"route": "SQL"}'])
    d = await LLMRouter(llm, H).route("anything")
    assert d.route is Route.SQL and d.source == "llm"


@pytest.mark.parametrize("reply", ["not json", '{"route": "graph"}', '{"x": 1}', "[1]"])
async def test_llm_router_falls_back_on_bad_reply(reply: str) -> None:
    d = await LLMRouter(FakeLLM([reply]), H).route("How many trades are there?")
    assert d.source == "heuristic" and d.route is Route.SQL


class BoomLLM(LLM):
    async def chat(
        self, messages: list[Message], *, json_mode: bool = False, max_tokens: int | None = None
    ) -> str:
        raise ConnectionError("ollama down")


class SlowLLM(LLM):
    async def chat(
        self, messages: list[Message], *, json_mode: bool = False, max_tokens: int | None = None
    ) -> str:
        await asyncio.sleep(1)
        return '{"route": "sql"}'


async def test_llm_router_falls_back_on_error_and_timeout() -> None:
    assert (await LLMRouter(BoomLLM(), H).route("policy?")).source == "heuristic"
    d = await LLMRouter(SlowLLM(), H, timeout_s=0.01).route("policy?")
    assert d.source == "heuristic"


async def test_cascade_skips_llm_when_confident() -> None:
    llm = FakeLLM(['{"route": "docs"}'])
    cascade = CascadeRouter(H, LLMRouter(llm, H), min_margin=3.0)
    d = await cascade.route("How many trades did EMEA clients make in AAPL?")
    assert d.source == "heuristic" and not llm.calls
    d = await cascade.route("Tell me something")
    assert d.source == "llm" and len(llm.calls) == 1
