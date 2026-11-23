from __future__ import annotations

from enterprise_rag.generation.answer import (
    REFUSAL,
    AnswerGenerator,
    format_sql_result,
    is_refusal,
    used_citations,
)
from enterprise_rag.retrieval.base import RetrievedChunk
from enterprise_rag.sql.executor import SQLResult
from tests.fakes import FakeLLM


def _chunk(i: int, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(i, f"doc{i}", 0, f"Doc {i}", f"content {i}", score)


CHUNKS = [_chunk(1), _chunk(2), _chunk(3)]


def test_used_citations_dedup_order_and_bounds() -> None:
    used = used_citations("A [2]. B [1][2]. C [9] [0].", CHUNKS)
    assert [c.document_id for c in used] == ["doc2", "doc1"]


def test_is_refusal() -> None:
    assert is_refusal(REFUSAL)
    assert is_refusal('"I do not know."')
    assert not is_refusal("The limit is 10 [1].")


async def test_no_chunks_refuses_without_llm() -> None:
    llm = FakeLLM([])
    out = await AnswerGenerator(llm).answer_docs("q", [])
    assert out.refused and out.text == REFUSAL and not llm.calls


async def test_low_score_refuses_without_llm() -> None:
    llm = FakeLLM([])
    out = await AnswerGenerator(llm, min_score=-2.0).answer_docs("q", [_chunk(1, -5.0)])
    assert out.refused and "threshold" in out.refusal_reason and not llm.calls


async def test_answer_with_citations_and_sources_in_prompt() -> None:
    llm = FakeLLM(["<think>hmm</think>The cap is 300 [3]."])
    out = await AnswerGenerator(llm, min_score=-2.0).answer_docs("hotel cap?", CHUNKS)
    assert out.text == "The cap is 300 [3]." and not out.refused
    assert [c.document_id for c in out.citations] == ["doc3"]
    prompt = llm.calls[0][1].content
    assert "[1] Doc 1" in prompt and "[3] Doc 3" in prompt and "hotel cap?" in prompt


async def test_model_refusal_is_normalised() -> None:
    out = await AnswerGenerator(FakeLLM(["I don't know."])).answer_docs("q", CHUNKS)
    assert out.refused and out.text == REFUSAL and out.citations == []


async def test_sql_answer_includes_rows() -> None:
    llm = FakeLLM(["There are 42 trades."])
    res = SQLResult(["n"], [[42]], 1, 1.0)
    out = await AnswerGenerator(llm).answer_sql("How many trades?", res)
    assert out.text == "There are 42 trades."
    assert "[42]" in llm.calls[0][1].content


async def test_hybrid_degrades_to_docs_when_sql_failed() -> None:
    llm = FakeLLM(["Limit is 5m [1]."])
    out = await AnswerGenerator(llm).answer_hybrid("q", CHUNKS, None)
    assert out.citations == [CHUNKS[0]]
    assert "Trading database" not in llm.calls[0][1].content


def test_format_sql_result_truncates() -> None:
    res = SQLResult(["a"], [[i] for i in range(30)], 30, 1.0)
    s = format_sql_result(res, max_rows=5)
    assert "25 more rows" in s and "(30 total)" in s
