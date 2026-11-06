"""Offline evaluation.

* Retrieval: Recall@5 and MRR@20 for vector / keyword / hybrid, each with and without the
  cross-encoder reranker, over 50 labelled doc questions.
* SQL: execution accuracy of Text2SQL on 30 questions with gold SQL.
* Router: accuracy over all 80 questions.

Relevance labels are evidence spans (see scripts/generate_docs.py); a chunk is relevant if
its whitespace-normalised text contains the span. Recall@5 is hit-rate: a question counts as
recalled if any relevant chunk is in the top 5 (every relevant chunk contains the whole
evidence, so one is sufficient to answer).

    python -m scripts.eval [--only retrieval|sql|router] [--no-rerank]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from sqlalchemy import text

from enterprise_rag.config import get_settings
from enterprise_rag.container import Container
from enterprise_rag.ingestion.embedder import Vector
from enterprise_rag.retrieval.base import RetrievedChunk, Retriever

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
RESULTS_MD = ROOT / "docs" / "eval_results.md"
CANDIDATES = 20
TOP_K = 5


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_jsonl(name: str) -> list[dict[str, Any]]:
    with (EVAL_DIR / name).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------------- retrieval ----
@dataclass
class RetrievalScore:
    name: str
    recall_at_5: float
    mrr: float
    p50_ms: float


async def resolve_relevant(c: Container, questions: list[dict[str, Any]]) -> dict[str, set[str]]:
    async with c.session_factory() as s:
        rows = (await s.execute(text("SELECT document_id, chunk_index, content FROM chunks"))).all()
    by_doc: dict[str, list[tuple[int, str]]] = {}
    for doc_id, idx, content in rows:
        by_doc.setdefault(doc_id, []).append((idx, norm(content)))
    relevant: dict[str, set[str]] = {}
    for q in questions:
        ev = norm(q["evidence"])
        keys = {f"{q['doc_id']}:{i}" for i, body in by_doc.get(q["doc_id"], []) if ev in body}
        if not keys:
            raise SystemExit(f"{q['id']}: evidence not found in any chunk of {q['doc_id']}")
        relevant[q["id"]] = keys
    return relevant


def score_ranking(ranked: list[str], relevant: set[str]) -> tuple[float, float]:
    recall = 1.0 if any(k in relevant for k in ranked[:TOP_K]) else 0.0
    rr = next((1.0 / i for i, k in enumerate(ranked, 1) if k in relevant), 0.0)
    return recall, rr


Search = Callable[[str, Vector], Awaitable[list[RetrievedChunk]]]


async def eval_search(
    name: str,
    search: Search,
    questions: list[dict[str, Any]],
    relevant: dict[str, set[str]],
    embeddings: dict[str, Vector],
) -> RetrievalScore:
    recalls, rrs, lat = [], [], []
    for q in questions:
        t0 = time.perf_counter()
        ranked = [ch.key for ch in await search(q["question"], embeddings[q["id"]])]
        lat.append((time.perf_counter() - t0) * 1000)
        r, rr = score_ranking(ranked, relevant[q["id"]])
        recalls.append(r)
        rrs.append(rr)
    lat.sort()
    return RetrievalScore(name, mean(recalls), mean(rrs), lat[len(lat) // 2])


def retrieval_variants(c: Container, rerank: bool) -> list[tuple[str, Search]]:
    retrievers: list[Retriever] = [c.vector, c.keyword, c.hybrid]
    variants: list[tuple[str, Search]] = []
    for r in retrievers:
        variants.append((r.name, lambda q, e, r=r: r.retrieve(q, CANDIDATES, embedding=e)))
    reranker = getattr(c, "reranker", None)
    if rerank and reranker is not None:
        for r in retrievers:

            async def reranked(q: str, e: Vector, r: Retriever = r) -> list[RetrievedChunk]:
                cands = await r.retrieve(q, CANDIDATES, embedding=e)
                return await reranker.rerank(q, cands, CANDIDATES)

            variants.append((f"{r.name} + rerank", reranked))
    return variants


async def run_retrieval(c: Container, rerank: bool) -> str:
    questions = load_jsonl("doc_questions.jsonl")
    relevant = await resolve_relevant(c, questions)
    # Embed every question once up front: latency columns then measure retrieval (and
    # rerank) only, and every variant sees the identical query vector.
    t0 = time.perf_counter()
    embeddings = {q["id"]: await c.embedder.embed_query(q["question"]) for q in questions}
    embed_ms = (time.perf_counter() - t0) * 1000 / len(questions)
    # Warm-up: load models / fill connection pool so latency numbers are steady-state.
    for _, search in retrieval_variants(c, rerank):
        await search(questions[0]["question"], embeddings[questions[0]["id"]])
    scores = [
        await eval_search(name, search, questions, relevant, embeddings)
        for name, search in retrieval_variants(c, rerank)
    ]
    lines = [
        f"### Retrieval ({len(questions)} labelled questions, {CANDIDATES} candidates)",
        "",
        "| Retriever | Recall@5 | MRR | p50 latency (ms) |",
        "|---|---:|---:|---:|",
    ]
    lines += [f"| {s.name} | {s.recall_at_5:.3f} | {s.mrr:.3f} | {s.p50_ms:.1f} |" for s in scores]
    lines += ["", f"Query embedding (Ollama, excluded above): {embed_ms:.1f} ms mean."]
    return "\n".join(lines)


# --------------------------------------------------------------------------------- main --
async def main_async(args: argparse.Namespace) -> None:
    c = Container.build(get_settings())
    sections: list[str] = []
    try:
        if args.only in (None, "retrieval"):
            sections.append(await run_retrieval(c, rerank=not args.no_rerank))
        for extra in EXTRA_SECTIONS:
            if args.only in (None, extra.__name__.removeprefix("run_")):
                sections.append(await extra(c))
    finally:
        await c.aclose()
    report = "\n\n".join(sections)
    print(report)
    if args.only is None:
        RESULTS_MD.parent.mkdir(exist_ok=True)
        RESULTS_MD.write_text(
            "# Evaluation results\n\nGenerated by `make eval`.\n\n" + report + "\n"
        )
        print(f"\nwrote {RESULTS_MD.relative_to(ROOT)}")


EXTRA_SECTIONS: list[Callable[[Container], Awaitable[str]]] = []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--only", choices=["retrieval", "sql", "router"])
    p.add_argument("--no-rerank", action="store_true")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
