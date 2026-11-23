"""Grounded answer generation with numbered citations.

The model sees the top-K reranked chunks as numbered sources ``[1] .. [K]`` and must cite
them inline. We then parse which numbers it actually used and return only those as
citations, so the UI never shows a source the answer does not rely on.

Two refusal paths keep the system honest:

* **pre-LLM**: if retrieval found nothing, or the best cross-encoder score is below
  ``min_score``, we refuse without calling the LLM (cheap, and avoids hallucinating from
  irrelevant context);
* **LLM**: the prompt instructs the model to reply with the refusal sentence when the
  sources do not contain the answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from enterprise_rag.llm.base import LLM
from enterprise_rag.retrieval.base import RetrievedChunk
from enterprise_rag.sql.executor import SQLResult

REFUSAL = "I don't know based on the available documents."

DOCS_PROMPT = f"""You are an assistant for employees of a financial firm.
Answer the question using ONLY the numbered sources below.

Rules:
- Cite every fact with its source number in square brackets, e.g. [1] or [2][3].
- Be concise: one to four sentences. Quote exact numbers, limits and names from the sources.
- If the sources do not contain the answer, reply exactly: "{REFUSAL}"
- Never use outside knowledge and never invent sources."""

SQL_PROMPT = """You explain database query results to business users at a financial firm.
Answer the question in one to three sentences using ONLY the result rows below.
Quote the exact numbers (round money to 2 decimals, use thousands separators).
If the result is empty, say that no matching records were found.
Do not describe the SQL itself."""

HYBRID_PROMPT = f"""You are an assistant for employees of a financial firm.
Answer the question by combining the database result and the numbered document sources.

Rules:
- Cite document facts with their source number in square brackets, e.g. [1].
- Refer to database figures as coming from "the trading database". Never put trade ids or
  other numbers in square brackets: brackets are reserved for source numbers.
- Be concise: two to five sentences. Quote exact numbers.
- If neither source answers the question, reply exactly: "{REFUSAL}"
- Never use outside knowledge."""

_CITE = re.compile(r"\[(\d{1,2})\]")


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    text: str
    citations: list[RetrievedChunk] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""


def format_sources(chunks: Sequence[RetrievedChunk], *, max_chars: int = 2400) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        body = c.content if len(c.content) <= max_chars else c.content[:max_chars] + " ..."
        parts.append(f"[{i}] {c.title}\n{body}")
    return "\n\n".join(parts)


def format_sql_result(result: SQLResult, *, max_rows: int = 25) -> str:
    rows = result.rows[:max_rows]
    lines = [json.dumps(result.columns)]
    lines += [json.dumps(r, default=str) for r in rows]
    if result.row_count > max_rows:
        lines.append(f"... ({result.row_count - max_rows} more rows not shown)")
    return f"columns + rows ({result.row_count} total):\n" + "\n".join(lines)


def used_citations(answer: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Chunks referenced as ``[n]`` in ``answer``, in first-mention order, deduplicated."""
    seen: list[int] = []
    for m in _CITE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(chunks) and n not in seen:
            seen.append(n)
    return [chunks[n - 1] for n in seen]


def is_refusal(answer: str) -> bool:
    a = answer.strip().strip('"').lower()
    return a.startswith("i don't know") or a.startswith("i do not know")


def _strip_think(text: str) -> str:
    # Reasoning models (qwen3, deepseek-r1) may prepend <think>...</think>.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class AnswerGenerator:
    def __init__(
        self,
        llm: LLM,
        *,
        min_score: float | None = None,
        max_tokens: int = 400,
    ) -> None:
        """``min_score`` is a cross-encoder logit threshold; ``None`` disables the pre-LLM
        refusal (use that when the reranker is a no-op and scores are RRF values)."""
        self._llm = llm
        self._min_score = min_score
        self._max_tokens = max_tokens

    def _pre_refusal(self, chunks: Sequence[RetrievedChunk]) -> GeneratedAnswer | None:
        if not chunks:
            return GeneratedAnswer(REFUSAL, refused=True, refusal_reason="no documents retrieved")
        best = max(c.score for c in chunks)
        if self._min_score is not None and best < self._min_score:
            return GeneratedAnswer(
                REFUSAL,
                refused=True,
                refusal_reason=f"best rerank score {best:.2f} < threshold {self._min_score:.2f}",
            )
        return None

    async def answer_docs(self, question: str, chunks: Sequence[RetrievedChunk]) -> GeneratedAnswer:
        if (refusal := self._pre_refusal(chunks)) is not None:
            return refusal
        user = f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"
        reply = _strip_think(
            await self._llm.complete(DOCS_PROMPT, user, max_tokens=self._max_tokens)
        )
        if is_refusal(reply):
            return GeneratedAnswer(REFUSAL, refused=True, refusal_reason="model refused")
        return GeneratedAnswer(reply, used_citations(reply, chunks))

    async def answer_sql(self, question: str, result: SQLResult) -> GeneratedAnswer:
        user = f"Question: {question}\n\nResult {format_sql_result(result)}"
        reply = _strip_think(await self._llm.complete(SQL_PROMPT, user, max_tokens=200))
        return GeneratedAnswer(reply)

    async def answer_hybrid(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        result: SQLResult | None,
    ) -> GeneratedAnswer:
        if result is None:  # SQL side failed: degrade to a documents-only answer
            return await self.answer_docs(question, chunks)
        db = f"Trading database result {format_sql_result(result)}"
        docs = format_sources(chunks) if chunks else "(no documents found)"
        user = f"{db}\n\nDocument sources:\n\n{docs}\n\nQuestion: {question}"
        reply = _strip_think(
            await self._llm.complete(HYBRID_PROMPT, user, max_tokens=self._max_tokens)
        )
        if is_refusal(reply):
            return GeneratedAnswer(REFUSAL, refused=True, refusal_reason="model refused")
        return GeneratedAnswer(reply, used_citations(reply, chunks))
