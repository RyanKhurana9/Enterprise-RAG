from __future__ import annotations

import json

import pytest

from enterprise_rag.sql.executor import SQLExecutionError, SQLResult
from enterprise_rag.sql.guard import GuardedSQL, SQLGuard
from enterprise_rag.sql.text2sql import Text2SQL, extract_sql
from tests.fakes import FakeLLM


class StubIntrospector:
    async def describe(self) -> str:
        return "TABLE trades (\n  id bigint PRIMARY KEY\n)"


class StubExecutor:
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.executed: list[str] = []

    async def execute(self, guarded: GuardedSQL) -> SQLResult:
        self.executed.append(guarded.sql)
        if self.fail_on and self.fail_on in guarded.sql:
            raise SQLExecutionError('column "nope" does not exist')
        return SQLResult(["n"], [[42]], 1, 1.0)


def make(llm: FakeLLM, executor: StubExecutor | None = None, repairs: int = 1) -> Text2SQL:
    return Text2SQL(
        llm,
        StubIntrospector(),  # type: ignore[arg-type]
        SQLGuard(["trades"]),
        executor or StubExecutor(),  # type: ignore[arg-type]
        repair_attempts=repairs,
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"sql": "SELECT 1;"}', "SELECT 1"),
        ("Here you go:\n```sql\nSELECT 2\n```", "SELECT 2"),
        ("```\nSELECT 3;\n```", "SELECT 3"),
        ("SELECT 4", "SELECT 4"),
        ('{"query": "x"}', '{"query": "x"}'),
    ],
)
def test_extract_sql(reply: str, expected: str) -> None:
    assert extract_sql(reply) == expected


async def test_happy_path_includes_schema_and_few_shots() -> None:
    llm = FakeLLM([json.dumps({"sql": "SELECT count(*) AS n FROM trades"})])
    res = await make(llm).run("how many trades?")
    assert res.ok and res.attempts == 1 and res.result is not None
    assert res.result.rows == [[42]]
    assert res.sql is not None and res.sql.endswith("LIMIT 1000")
    system = llm.calls[0][0].content
    assert "TABLE trades" in system and "notional = qty * price" in system
    assert llm.calls[0][-1].content == "how many trades?"
    assert len(llm.calls[0]) > 3  # few-shot turns present


async def test_guard_rejection_triggers_repair_with_error_feedback() -> None:
    llm = FakeLLM(['{"sql": "DELETE FROM trades"}', '{"sql": "SELECT 1 FROM trades"}'])
    res = await make(llm).run("q")
    assert res.ok and res.attempts == 2
    assert "only SELECT" in res.errors[0]
    assert "That query failed" in llm.calls[1][-1].content


async def test_execution_error_triggers_repair() -> None:
    ex = StubExecutor(fail_on="nope")
    llm = FakeLLM(['{"sql": "SELECT nope FROM trades"}', '{"sql": "SELECT id FROM trades"}'])
    res = await make(llm, ex).run("q")
    assert res.ok and len(ex.executed) == 2 and "does not exist" in res.errors[0]


async def test_gives_up_after_repair_budget() -> None:
    llm = FakeLLM(['{"sql": "DROP TABLE trades"}'] * 3)
    res = await make(llm, repairs=1).run("q")
    assert not res.ok and res.attempts == 2 and len(res.errors) == 2
