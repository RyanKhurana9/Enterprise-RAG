"""Prometheus metrics (exposed at ``GET /metrics``)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80)

QUERIES = Counter("rag_queries_total", "Answered queries", ["route", "cache", "outcome"])
QUERY_LATENCY = Histogram(
    "rag_query_seconds", "End-to-end query latency", ["route", "cache"], buckets=_BUCKETS
)
STAGE_LATENCY = Histogram("rag_stage_seconds", "Per-stage latency", ["stage"], buckets=_BUCKETS)
