"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).

    score(d) = Σ_r  w_r / (k + rank_r(d))          rank is 1-based

RRF only uses ranks, never raw scores, so it fuses cosine similarities (0..1) and
``ts_rank_cd`` values (unbounded, corpus-dependent) without any score calibration. ``k``
dampens the advantage of top ranks; 60 is the value from the original paper and is robust.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[K]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[K, float]]:
    """Fuse ranked lists of ids. Returns ``(id, score)`` sorted by score descending.

    Ties are broken by the order in which an id is first seen (list order, then rank), so the
    output is deterministic. Duplicate ids within one ranking count only at their best rank.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")

    scores: dict[K, float] = {}
    first_seen: dict[K, int] = {}
    order = 0
    for ranking, w in zip(rankings, weights, strict=True):
        seen_here: set[K] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen_here:
                continue
            seen_here.add(item)
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
