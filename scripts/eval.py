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
import datetime as dt
import json
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any

from sqlalchemy import text

from enterprise_rag.config import get_settings
from enterprise_rag.container import Container
from enterprise_rag.ingestion.embedder import Vector
from enterprise_rag.retrieval.base import RetrievedChunk, Retriever
from enterprise_rag.routing.router import CascadeRouter, QueryRouter
from enterprise_rag.sql.executor import SQLExecutor

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


# ------------------------------------------------------------------------------- SQL ----
def _norm_value(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int | float | Decimal):
        return round(float(v), 2)
    if isinstance(v, dt.datetime | dt.date):
        return v.isoformat()[:10] if isinstance(v, dt.date) else v.isoformat()
    if isinstance(v, str):
        # ISO timestamps from JSON-ified results compare at date precision
        return v[:10] if re.fullmatch(r"\d{4}-\d{2}-\d{2}T.*", v) else v
    return v


def _columns(rows: list[list[Any]]) -> list[Counter[Any]]:
    if not rows:
        return []
    return [Counter(_norm_value(r[i]) for r in rows) for i in range(len(rows[0]))]


def results_match(gold: list[list[Any]], pred: list[list[Any]]) -> tuple[bool, bool]:
    """Return ``(strict, relaxed)`` execution-match flags.

    * strict: same multiset of rows, comparing each row as an unordered bag of values
      (column order and aliases do not matter; numbers compared at 2dp).
    * relaxed: same row count and every gold column appears (as a value multiset) among
      the predicted columns – tolerates extra columns such as an added ``id`` or label.
    """
    g = Counter(tuple(sorted(map(repr, map(_norm_value, r)))) for r in gold)
    p = Counter(tuple(sorted(map(repr, map(_norm_value, r)))) for r in pred)
    strict = g == p
    if len(gold) != len(pred):
        return strict, False
    pred_cols = _columns(pred)
    relaxed = all(any(gc == pc for pc in pred_cols) for gc in _columns(gold))
    return strict, strict or relaxed


async def run_sql(c: Container) -> str:
    questions = load_jsonl("sql_questions.jsonl")
    strict_hits, relaxed_hits, repaired, failures, lat = 0, 0, 0, [], []
    for q in questions:
        guarded = c.guard.validate(q["gold_sql"])
        gold = await SQLExecutor(c.readonly_engine).execute(guarded)
        t0 = time.perf_counter()
        res = await c.text2sql.run(q["question"])
        lat.append((time.perf_counter() - t0) * 1000)
        repaired += res.attempts > 1 and res.ok
        if res.result is None:
            failures.append(f"| {q['id']} | error: {res.errors[-1][:80]} |")
            continue
        strict, relaxed = results_match(gold.rows, res.result.rows)
        strict_hits += strict
        relaxed_hits += relaxed
        if not relaxed:
            failures.append(f"| {q['id']} | `{(res.sql or '')[:120]}` |")
    n = len(questions)
    lat.sort()
    lines = [
        f"### Text2SQL ({n} questions, model `{c.settings.llm_model}`)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Execution accuracy (relaxed: gold columns ⊆ predicted) | {relaxed_hits / n:.3f} |",
        f"| Execution accuracy (strict: identical result set) | {strict_hits / n:.3f} |",
        f"| Queries fixed by the repair loop | {repaired} |",
        f"| p50 / max latency (ms) | {lat[n // 2]:.0f} / {lat[-1]:.0f} |",
    ]
    if failures:
        lines += ["", "Mismatches:", "", "| id | predicted SQL / error |", "|---|---|", *failures]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- router ----
async def run_router(c: Container) -> str:
    questions = load_jsonl("doc_questions.jsonl") + load_jsonl("sql_questions.jsonl")
    lines = [
        f"### Router ({len(questions)} questions: 50 docs + 30 sql)",
        "",
        "| Router | Accuracy | docs→sql errors | sql→docs errors | any→hybrid | p50 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    routers: list[tuple[str, QueryRouter]] = [
        ("LLM (heuristic fallback)", c.llm_router),
        ("cascade (LLM only if ambiguous)", CascadeRouter(c.heuristic_router, c.llm_router)),
        ("heuristic only", c.heuristic_router),
    ]
    for name, router in routers:
        confusion: Counter[tuple[str, str]] = Counter()
        lat = []
        for q in questions:
            t0 = time.perf_counter()
            decision = await router.route(q["question"])
            lat.append((time.perf_counter() - t0) * 1000)
            confusion[(q["route"], decision.route.value)] += 1
        correct = sum(v for (gold, pred), v in confusion.items() if gold == pred)
        hybrid = sum(v for (_, pred), v in confusion.items() if pred == "hybrid")
        lat.sort()
        lines.append(
            f"| {name} | {correct / len(questions):.3f} | {confusion[('docs', 'sql')]} | "
            f"{confusion[('sql', 'docs')]} | {hybrid} | {lat[len(lat) // 2]:.1f} |"
        )
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


EXTRA_SECTIONS: list[Callable[[Container], Awaitable[str]]] = [run_sql, run_router]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--only", choices=["retrieval", "sql", "router"])
    p.add_argument("--no-rerank", action="store_true")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
