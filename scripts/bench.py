"""Load-test the running API.

    python -m scripts.bench [--url http://localhost:8000] [--n 40] [--concurrency 4]
                            [--endpoint query|retrieve] [--no-cache]

Questions are sampled from the eval sets. Reports throughput, p50/p95/max latency, the
cache-hit mix and errors. Run it twice to see the cache's effect (the second run is
served almost entirely from the exact cache).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"


def questions() -> list[str]:
    out = []
    for name in ("doc_questions.jsonl", "sql_questions.jsonl"):
        with (EVAL_DIR / name).open() as fh:
            out += [json.loads(line)["question"] for line in fh if line.strip()]
    return out


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


async def run(args: argparse.Namespace) -> None:
    qs = questions()
    random.Random(0).shuffle(qs)
    qs = (qs * (args.n // len(qs) + 1))[: args.n]
    sem = asyncio.Semaphore(args.concurrency)
    lat: list[float] = []
    kinds: Counter[str] = Counter()
    errors: list[str] = []

    async with httpx.AsyncClient(base_url=args.url, timeout=300) as client:

        async def one(q: str) -> None:
            async with sem:
                t0 = time.perf_counter()
                try:
                    if args.endpoint == "query":
                        body = {"query": q, "use_cache": not args.no_cache}
                        r = await client.post("/query", json=body)
                    else:
                        r = await client.post("/retrieve", json={"query": q, "k": 5})
                    r.raise_for_status()
                    data = r.json()
                    kinds[
                        f"{data.get('route', {}).get('route', '-')}/{data.get('cache', '-')}"
                    ] += 1
                except Exception as e:
                    errors.append(f"{type(e).__name__}: {e}"[:160])
                lat.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        await asyncio.gather(*(one(q) for q in qs))
        wall = time.perf_counter() - t0

    print(f"{args.n} requests to /{args.endpoint}, concurrency {args.concurrency}")
    print(f"throughput {args.n / wall:.2f} req/s, wall {wall:.1f}s")
    print(f"latency ms  p50 {pct(lat, 50):.0f}  p95 {pct(lat, 95):.0f}  max {max(lat):.0f}")
    if args.endpoint == "query":
        print("route/cache:", dict(kinds.most_common()))
    print(f"errors: {len(errors)}")
    for e in errors[:5]:
        print("  ", e)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--endpoint", choices=["query", "retrieve"], default="query")
    p.add_argument("--no-cache", action="store_true")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
