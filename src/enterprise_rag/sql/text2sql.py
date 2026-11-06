"""Text2SQL: schema-grounded, few-shot prompting with a bounded self-repair loop."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from enterprise_rag.llm.base import LLM, Message
from enterprise_rag.sql.executor import SQLExecutionError, SQLExecutor, SQLResult
from enterprise_rag.sql.guard import SQLGuard, SQLGuardError
from enterprise_rag.sql.introspector import SchemaIntrospector

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You translate business questions into a single PostgreSQL SELECT query.

Schema:
{schema}

Rules:
- Use only the tables and columns in the schema. Never modify data.
- Write exactly one SELECT statement. No comments, no semicolons.
- Use explicit JOIN ... ON with the foreign keys shown.
- Trade notional = qty * price.
- Filter dates with half-open ranges, e.g. traded_at >= '2024-01-01' AND traded_at < '2025-01-01'.
- Only filter by date when the question names a time period.
- For "top N" / "most" / "highest" questions use ORDER BY ... DESC LIMIT N.
- Use the exact categorical values listed in the schema (case-sensitive).
- Give aggregate columns short snake_case aliases.

Respond with JSON only: {{"sql": "<query>"}}"""


@dataclass(frozen=True, slots=True)
class FewShot:
    question: str
    sql: str


DEFAULT_FEW_SHOTS: tuple[FewShot, ...] = (
    FewShot(
        "How many clients are in the Retail segment?",
        "SELECT count(*) AS client_count FROM clients WHERE segment = 'Retail'",
    ),
    FewShot(
        "What is the total notional traded in JPM during 2023?",
        "SELECT sum(qty * price) AS total_notional FROM trades WHERE ticker = 'JPM' "
        "AND traded_at >= '2023-01-01' AND traded_at < '2024-01-01'",
    ),
    FewShot(
        "Which 3 regions have the most accounts?",
        "SELECT c.region, count(*) AS account_count FROM accounts a "
        "JOIN clients c ON c.id = a.client_id GROUP BY c.region "
        "ORDER BY account_count DESC LIMIT 3",
    ),
    FewShot(
        "Show the number of trades per month in 2023 for Technology stocks.",
        "SELECT date_trunc('month', t.traded_at) AS month, count(*) AS trade_count "
        "FROM trades t JOIN instruments i ON i.ticker = t.ticker "
        "WHERE i.sector = 'Technology' AND t.traded_at >= '2023-01-01' "
        "AND t.traded_at < '2024-01-01' GROUP BY month ORDER BY month",
    ),
    FewShot(
        "Which clients have more than 500 trades?",
        "SELECT c.id, c.name, count(*) AS trade_count FROM clients c "
        "JOIN accounts a ON a.client_id = c.id JOIN trades t ON t.account_id = a.id "
        "GROUP BY c.id, c.name HAVING count(*) > 500 ORDER BY trade_count DESC",
    ),
)


@dataclass
class Text2SQLResult:
    question: str
    sql: str | None
    result: SQLResult | None
    attempts: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.result is not None


_FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.S | re.I)


def extract_sql(reply: str) -> str:
    """Pull SQL out of a JSON object, a fenced block, or bare text (in that order)."""
    reply = reply.strip()
    try:
        obj = json.loads(reply)
        if isinstance(obj, dict) and isinstance(obj.get("sql"), str):
            return str(obj["sql"]).strip().rstrip(";").strip()
    except json.JSONDecodeError:
        pass
    if m := _FENCE.search(reply):
        return m.group(1).strip().rstrip(";").strip()
    return reply.rstrip(";").strip()


class Text2SQL:
    def __init__(
        self,
        llm: LLM,
        introspector: SchemaIntrospector,
        guard: SQLGuard,
        executor: SQLExecutor,
        *,
        few_shots: Sequence[FewShot] = DEFAULT_FEW_SHOTS,
        repair_attempts: int = 1,
    ) -> None:
        self._llm = llm
        self._introspector = introspector
        self._guard = guard
        self._executor = executor
        self._few_shots = list(few_shots)
        self._repairs = repair_attempts

    async def _messages(self, question: str) -> list[Message]:
        schema = await self._introspector.describe()
        msgs = [Message("system", SYSTEM_PROMPT.format(schema=schema))]
        for fs in self._few_shots:
            msgs.append(Message("user", fs.question))
            msgs.append(Message("assistant", json.dumps({"sql": fs.sql})))
        msgs.append(Message("user", question))
        return msgs

    async def run(self, question: str) -> Text2SQLResult:
        messages = await self._messages(question)
        out = Text2SQLResult(question, None, None, 0)
        for attempt in range(1 + self._repairs):
            out.attempts = attempt + 1
            reply = await self._llm.chat(messages, json_mode=True)
            candidate = extract_sql(reply)
            out.sql = candidate
            try:
                guarded = self._guard.validate(candidate)
                out.sql = guarded.sql
                out.result = await self._executor.execute(guarded)
                return out
            except (SQLGuardError, SQLExecutionError) as e:
                err = str(e)
                log.info("text2sql attempt %d failed: %s", attempt + 1, err)
                out.errors.append(err)
                # Self-repair: show the model its own query and the error, ask again.
                messages = [
                    *messages,
                    Message("assistant", json.dumps({"sql": candidate})),
                    Message(
                        "user",
                        f"That query failed: {err}\nReturn a corrected query as JSON.",
                    ),
                ]
        return out
