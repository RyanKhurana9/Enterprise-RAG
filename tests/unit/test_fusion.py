from __future__ import annotations

import pytest

from enterprise_rag.retrieval.fusion import reciprocal_rank_fusion


def test_single_list_scores_are_1_over_k_plus_rank() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert [x for x, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1 / 61)
    assert fused[2][1] == pytest.approx(1 / 63)


def test_scores_sum_across_lists() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_consensus_beats_single_top_rank() -> None:
    # "x" is 2nd in both lists, "a" and "b" each top one list only.
    fused = reciprocal_rank_fusion([["a", "x"], ["b", "x"]], k=60)
    assert fused[0][0] == "x"


def test_ties_broken_by_first_appearance() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    assert fused[0][1] == fused[1][1]
    assert [x for x, _ in fused] == ["a", "b"]


def test_k_controls_top_rank_advantage() -> None:
    small = dict(reciprocal_rank_fusion([["a", "b"]], k=0))
    large = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    assert small["a"] / small["b"] == pytest.approx(2.0)
    assert large["a"] / large["b"] == pytest.approx(1002 / 1001)


def test_weights_and_duplicates() -> None:
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[2.0, 1.0]))
    assert fused["a"] == pytest.approx(2 / 61)
    dup = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=60))
    assert dup["a"] == pytest.approx(1 / 61) and dup["b"] == pytest.approx(1 / 63)


def test_empty_and_invalid() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)
