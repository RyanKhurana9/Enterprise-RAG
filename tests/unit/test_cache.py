from __future__ import annotations

import fakeredis
import numpy as np
import pytest

from enterprise_rag.cache.response_cache import (
    ResponseCache,
    key_tokens,
    normalize_query,
    query_hash,
)


def test_normalize_and_hash_ignore_case_space_and_trailing_punct() -> None:
    assert normalize_query("  How many   Trades? ") == "how many trades"
    assert query_hash("How many trades?") == query_hash("how many  trades")


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("trades in 2023", "trades in 2024", False),
        ("total for AAPL", "total for MSFT", False),
        ("limit above 250,000", "limit above 250000", True),
        ("How many trades in EMEA?", "how many EMEA trades", True),
        ("travel policy hotel cap", "hotel cap in travel policy", True),
    ],
)
def test_key_tokens(a: str, b: str, same: bool) -> None:
    assert (key_tokens(a) == key_tokens(b)) is same


async def test_exact_roundtrip_and_clear() -> None:
    cache = ResponseCache(fakeredis.FakeAsyncRedis(), None)
    assert await cache.get("Q?", None) is None
    await cache.put("Q?", None, {"answer": "A"})
    hit = await cache.get("  q ", None)
    assert hit is not None and hit.kind == "exact" and hit.response == {"answer": "A"}
    assert (await cache.stats())["exact_entries"] == 1
    assert await cache.clear() == 1
    assert await cache.get("Q?", None) is None


class _BrokenRedis:
    async def get(self, *_: object) -> None:
        raise ConnectionError("down")

    async def set(self, *_: object, **__: object) -> None:
        raise ConnectionError("down")


async def test_redis_errors_are_misses_not_failures() -> None:
    cache = ResponseCache(_BrokenRedis(), None)  # type: ignore[arg-type]
    await cache.put("q", np.zeros(4, dtype=np.float32), {"a": 1})
    assert await cache.get("q", None) is None
