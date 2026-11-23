# Enterprise-RAG

Question answering over **unstructured documents** (internal policies, 10-K filings, employee
FAQs) **and a structured trading warehouse**, behind one API and a Streamlit UI. Everything
runs locally: one Postgres (pgvector + full-text search), Redis, and Ollama.

```
                 ┌──────────────── Streamlit UI (:8501) ────────────────┐
                 │ chat · citations · SQL + charts · retrieval inspector │
                 └──────────────────────────┬───────────────────────────┘
                                            │ HTTP
                                  FastAPI (:8000)  /query /retrieve /health /metrics
                                            │
      exact cache (Redis) ─► semantic cache (pgvector, number-guarded) ─► router (LLM | heuristic | cascade)
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
          docs                            hybrid                           sql
  vector (HNSW) ∥ keyword (GIN)       docs ∥ sql            Text2SQL (schema + few-shot, self-repair)
        └── RRF fusion ──┘                │                   SQLGuard ─► read-only role, READ ONLY txn,
   cross-encoder rerank 20 → 5            │                   statement_timeout
   refuse if best score < threshold       │                          │
        LLM answer with [n] citations ◄───┴──────────────► LLM summary of result rows
```

## Quick start

Prerequisites: Docker, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com)
running on the host with the two models pulled:

```bash
ollama pull llama3.1:8b && ollama pull nomic-embed-text

make install          # .venv with dev, rerank and ui extras
docker compose up -d postgres redis
make seed             # migrations + 100k synthetic trades
make ingest           # chunk + embed the 202 documents in data/docs (idempotent)
make api              # FastAPI on http://localhost:8000  (docs at /docs)
make ui               # Streamlit on http://localhost:8501
```

Or run the whole stack in containers (API and UI included; still uses the host's Ollama,
which is much faster than Docker on macOS because it has Metal GPU access):

```bash
make up               # postgres, redis, api, ui
make up-ollama        # same, but with Ollama in a container too
```

## The UI

`ui/streamlit_app.py` is a thin client of the API (`RAG_API_URL`, default
`http://localhost:8000`):

- **Ask**: a chat with example questions per source. Each answer shows route, router
  source, cache status (exact / semantic + similarity) and latency badges, the cited and
  retrieved sources with rerank scores, and the generated SQL with its result table and an
  automatic chart. An optional debug panel shows per-stage timings.
- **Retrieval inspector**: vector, keyword and hybrid results side by side, with or
  without the cross-encoder.
- **Sidebar**: live dependency health, corpus and cache stats, the active configuration,
  route override, cache toggle, top-k, and buttons to clear the cache or re-ingest.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/query` | `{"query": "...", "route"?: "docs\|sql\|hybrid", "use_cache"?: true, "top_k"?: 5}` |
| `POST` | `/retrieve` | `{"query": "...", "mode": "vector\|keyword\|hybrid", "rerank": true, "k": 5}` |
| `POST` | `/ingest` | re-ingest `data/docs` (unchanged documents are skipped) |
| `GET` | `/health` | probes postgres / redis / ollama; `503` only when a critical dependency is down |
| `GET` | `/stats` | document, chunk and cache counts plus the active config |
| `DELETE` | `/cache` | clear both cache tiers |
| `GET` | `/metrics` | Prometheus: `rag_queries_total`, `rag_query_seconds`, `rag_stage_seconds` |

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query": "Which 5 tickers have the highest total notional?"}' | jq .answer
```

## Design notes

- **Hybrid retrieval**: dense (nomic-embed-text with task prefixes, HNSW cosine) and keyword
  (OR-ed lexemes, `ts_rank_cd` over a GIN index) run concurrently, embedded once, and are
  fused with Reciprocal Rank Fusion (k = 60).
- **Reranking**: a cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores the top 20 down to 5.
  Its raw logits are comparable across queries, so the generator **refuses without calling
  the LLM** when the best chunk scores below `RAG_MIN_RERANK_SCORE`.
- **Grounded generation**: sources are numbered `[1]..[k]`, and the model must cite them.
  Only sources actually cited are flagged in the response.
- **Text2SQL safety, in three layers**: (1) `SQLGuard` allows only a single `SELECT` over
  allow-listed tables, with no comments, DML/DDL, `SELECT INTO`, locks or dangerous
  `pg_*`/`dblink` functions, and enforces a `LIMIT`; (2) execution in a `READ ONLY`
  transaction with a `statement_timeout`; (3) a Postgres role that only has `SELECT` on the
  four finance tables.
- **Two-tier cache**: an exact match in Redis (normalized query hash), then a semantic match
  in Postgres (cosine ≥ `RAG_SEMANTIC_CACHE_THRESHOLD`). A semantic hit also requires the
  same numbers and upper-case codes, so "trades in 2023" is never served the 2024 answer.
  The cache is best-effort: errors count as misses.
- **Routing**: an LLM classifier with a heuristic fallback (default), a heuristic only, or a
  cascade (heuristic unless the keyword margin is small).

Every threshold is configurable via `RAG_*` environment variables (see
`src/enterprise_rag/config.py`).

## Evaluation

`make eval` runs offline evaluation over the labeled sets in `data/eval` and writes
`docs/eval_results.md`. Last recorded results (llama3.1:8b, nomic-embed-text):

| | |
|---|---|
| Retrieval Recall@5 / MRR, hybrid + rerank (50 Qs) | 0.96 / 0.845 |
| Retrieval Recall@5 / MRR, hybrid without rerank | 0.90 / 0.806 |
| Text2SQL execution accuracy, relaxed / strict (30 Qs) | 0.833 / 0.700 |
| Router accuracy: LLM / cascade / heuristic (80 Qs) | 1.000 / 1.000 / 0.988 |

## Operations

```bash
make health           # scripts/healthcheck.sh
make explain          # EXPLAIN ANALYZE of the retrieval and cache queries (--force-index)
make bench            # load test: python -m scripts.bench --n 40 --concurrency 4
```

## Development

```bash
make check            # ruff lint + format check, mypy --strict, pytest with coverage
make test-unit        # unit tests only (no Docker)
```

Integration tests run against a throwaway pgvector Postgres via testcontainers and are
skipped when Docker is unavailable.

## Layout

```
src/enterprise_rag/
  api/app.py            FastAPI app factory
  service.py            end-to-end orchestration (cache → route → retrieve/SQL → generate)
  generation/answer.py  grounded prompts, citations, refusals
  cache/                exact (Redis) + semantic (pgvector) response cache
  retrieval/            vector, keyword, hybrid (RRF), cross-encoder reranker
  sql/                  Text2SQL, SQLGuard, read-only executor, schema introspector
  routing/router.py     LLM / heuristic / cascade routers
  ingestion/            loaders (md, pdf), token-aware chunker, embedder, idempotent pipeline
  container.py          composition root (the only place that knows concrete classes)
ui/streamlit_app.py     Streamlit front-end
scripts/                seed, ingest, eval, explain, bench, generate_docs
migrations/             Alembic schema (finance + RAG tables, read-only role)
```
