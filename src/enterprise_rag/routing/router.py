"""Query routing: docs (unstructured RAG) | sql (warehouse) | hybrid (both)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from enterprise_rag.llm.base import LLM

log = logging.getLogger(__name__)


class Route(StrEnum):
    DOCS = "docs"
    SQL = "sql"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: Route
    source: str  # "llm" | "heuristic"
    reason: str = ""
    margin: float = 0.0  # heuristic only: |sql score - docs score|


class QueryRouter(ABC):
    @abstractmethod
    async def route(self, query: str) -> RouteDecision: ...


# Signals are (pattern, weight). Data-entity nouns are strong SQL evidence; aggregation
# phrasing alone is weak because doc questions also ask "how many days ...".
_SQL_SIGNALS: tuple[tuple[str, float], ...] = (
    (r"\btrades?\b|\btraded\b|\btrading volume\b", 2.0),
    (r"\bclients?\b", 1.0),
    (r"\baccounts?\b", 1.0),
    (r"\bnotional\b|\bquantity\b|\bqty\b|\bshares (?:bought|sold)\b", 2.0),
    (r"\btickers?\b|\binstruments?\b|\basset class\b|\bsectors?\b", 1.5),
    (r"\b(?:buy|sell) trades\b|\bside\b", 1.5),
    (r"\b(?:AMER|EMEA|APAC|LATAM)\b", 2.0),
    (r"\b(?:Retail|Institutional|Private Wealth|Corporate) (?:segment|clients?)\b", 2.0),
    (r"\b(?:EUR|GBP|JPY|CHF|USD)\b.*\baccounts?\b|\bdenominated\b", 1.5),
    (r"\bhow many\b|\bcount\b|\bnumber of\b", 0.5),
    (r"\btotal\b|\bsum\b|\baverage\b|\bavg\b|\bmaximum\b|\bminimum\b", 0.5),
    (r"\btop \d+\b|\bper (?:month|region|segment|account|ticker|sector)\b|\beach\b", 1.0),
    (r"\b(?:AAPL|MSFT|NVDA|GOOGL|META|AMZN|TSLA|JPM|GS|SPY|QQQ|GLD|TLT|XOM)\b", 2.0),
)
_DOC_SIGNALS: tuple[tuple[str, float], ...] = (
    (r"\bpolic(?:y|ies)\b|\bprocedure\b|\bguideline\b|\brules?\b", 2.0),
    (r"\bfaq\b|\bhandbook\b|\bintranet\b", 2.0),
    (r"\b10-?k\b|\bannual report\b|\bfiscal\b|\bheadquarter", 2.0),
    (r"\b(?:Global Markets|Wealth Management|Corporate Banking)\b", 2.0),
    (r"\bemployees?\b|\bstaff\b|\bheadcount\b", 1.5),
    (r"\bmust\b|\brequire[sd]?\b|\ballowed\b|\bpermitted\b|\bmandatory\b", 1.0),
    (r"\bapprov\w*\b|\bsign off\b|\bescalat\w*\b|\bdeadline\b", 1.0),
    (r"\b(?:Technologies|Industries|Brands|Health|Financial|Energy)\b", 1.5),
    (r"\bwho\b|\bwhy\b|\bexplain\b|\bdescribe\b", 0.5),
    (r"\bhow (?:long|often|soon|quickly|frequently|fast)\b", 1.0),
)


def _score(query: str, signals: tuple[tuple[str, float], ...]) -> float:
    return sum(w for pat, w in signals if re.search(pat, query, flags=re.I))


class HeuristicRouter(QueryRouter):
    """Weighted keyword scoring. Zero-latency and always available."""

    def __init__(self, *, hybrid_ratio: float = 0.6, min_score: float = 2.0) -> None:
        self._ratio = hybrid_ratio
        self._min = min_score

    async def route(self, query: str) -> RouteDecision:
        return self.route_sync(query)

    def route_sync(self, query: str) -> RouteDecision:
        sql, docs = _score(query, _SQL_SIGNALS), _score(query, _DOC_SIGNALS)
        reason = f"sql={sql:.1f} docs={docs:.1f}"
        margin = abs(sql - docs)
        # Hybrid: both sources clearly signalled and neither dominates.
        if min(sql, docs) >= self._min and min(sql, docs) / max(sql, docs) >= self._ratio:
            return RouteDecision(Route.HYBRID, "heuristic", reason, margin)
        if sql > docs:
            return RouteDecision(Route.SQL, "heuristic", reason, margin)
        return RouteDecision(Route.DOCS, "heuristic", reason, margin)


ROUTER_PROMPT = """Classify which data source can answer a question at a financial firm.

- "sql": the question asks for numbers computed from the trading database. Tables:
  clients(name, region AMER/EMEA/APAC/LATAM, segment), accounts(currency, opened_at),
  trades(ticker, side BUY/SELL, qty, price, traded_at), instruments(ticker, asset_class, sector).
  Counts, totals, averages, rankings of trades, clients, accounts or instruments.
- "docs": the answer is written in a document: internal policies (limits, thresholds,
  approvals, deadlines for divisions such as Global Markets, Wealth Management, Corporate
  Banking), company 10-K filings (revenue, employees, headquarters, debt, dividends), or
  employee FAQs (payroll, benefits, leave, IT, onboarding).
- "hybrid": needs both, e.g. comparing trading data against a policy limit.

Examples:
Q: How many trades did EMEA clients make in 2024? -> sql
Q: What is the hotel cap in the Wealth Management travel policy? -> docs
Q: What was Fairview Holdings' revenue in fiscal 2023? -> docs
Q: Which ticker has the highest total notional? -> sql
Q: Do any single trades exceed the Global Markets VaR limit? -> hybrid

Respond with JSON only: {"route": "sql" | "docs" | "hybrid"}"""


class LLMRouter(QueryRouter):
    """LLM zero/few-shot classifier; falls back to the heuristic on error, timeout or an
    unparseable reply, so routing never fails a request."""

    def __init__(self, llm: LLM, fallback: QueryRouter, *, timeout_s: float = 20.0) -> None:
        self._llm = llm
        self._fallback = fallback
        self._timeout = timeout_s

    async def route(self, query: str) -> RouteDecision:
        try:
            reply = await asyncio.wait_for(
                # A one-field JSON answer needs ~8 tokens; capping output bounds latency.
                self._llm.complete(ROUTER_PROMPT, f"Q: {query}", json_mode=True, max_tokens=16),
                timeout=self._timeout,
            )
            obj = json.loads(reply)
            route = Route(str(obj["route"]).strip().lower())
            return RouteDecision(route, "llm", "llm classification")
        except (TimeoutError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            log.info("LLM router fell back to heuristic: %r", e)
        except Exception as e:  # network errors etc.
            log.warning("LLM router error, falling back: %r", e)
        return await self._fallback.route(query)


class CascadeRouter(QueryRouter):
    """Heuristic first; consult the LLM only when the keyword margin is small.

    Most production traffic is unambiguous, so this keeps the LLM's accuracy on hard
    queries while skipping a ~1s LLM call on the easy majority.
    """

    def __init__(
        self, heuristic: HeuristicRouter, llm_router: QueryRouter, *, min_margin: float = 3.0
    ) -> None:
        self._heuristic = heuristic
        self._llm = llm_router
        self._min_margin = min_margin

    async def route(self, query: str) -> RouteDecision:
        decision = self._heuristic.route_sync(query)
        if decision.margin >= self._min_margin and decision.route is not Route.HYBRID:
            return decision
        return await self._llm.route(query)
