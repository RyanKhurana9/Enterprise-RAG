from __future__ import annotations

import pytest

from enterprise_rag.sql.guard import SQLGuard, SQLGuardError

GUARD = SQLGuard(["clients", "accounts", "trades", "instruments"], max_rows=1000)


# ---------------------------------------------------------------------- accepted ----
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM trades",
        "select t.ticker, sum(t.qty * t.price) from trades t join instruments i "
        "on i.ticker = t.ticker group by 1 order by 2 desc",
        "WITH x AS (SELECT client_id FROM accounts) SELECT count(*) FROM x",
        "SELECT ticker FROM trades UNION SELECT ticker FROM instruments",
        "SELECT * FROM public.clients",
        "SELECT 'DROP TABLE trades; --' AS harmless_literal FROM clients",
        "SELECT count(*) FROM trades;",  # single trailing semicolon is fine
        "SELECT * FROM generate_series(1, 3) AS g",
    ],
)
def test_accepts_read_only_queries(sql: str) -> None:
    GUARD.validate(sql)


def test_limit_added_when_missing() -> None:
    g = GUARD.validate("SELECT * FROM trades")
    assert g.limit_applied and g.sql.endswith("LIMIT 1000")


def test_limit_clamped_when_too_large() -> None:
    g = GUARD.validate("SELECT * FROM trades LIMIT 50000")
    assert g.sql.endswith("LIMIT 1000") and "50000" not in g.sql


def test_small_limit_preserved() -> None:
    g = GUARD.validate("SELECT * FROM trades ORDER BY id LIMIT 5")
    assert not g.limit_applied and g.sql.endswith("LIMIT 5")


def test_non_literal_limit_replaced() -> None:
    g = GUARD.validate("SELECT * FROM trades LIMIT (SELECT 100000)")
    assert g.sql.endswith("LIMIT 1000")


def test_union_is_wrapped_and_limited() -> None:
    g = GUARD.validate("SELECT ticker FROM trades UNION ALL SELECT ticker FROM instruments")
    assert g.sql.startswith("SELECT * FROM (") and g.sql.endswith("LIMIT 1000")


def test_reports_tables() -> None:
    g = GUARD.validate("SELECT * FROM trades t JOIN accounts a ON a.id = t.account_id")
    assert g.tables == {"trades", "accounts"}


# ---------------------------------------------------------------------- rejected ----
@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DROP TABLE trades", "only SELECT"),
        ("drop table if exists clients cascade", "only SELECT"),
        ("UPDATE trades SET qty = 0", "only SELECT"),
        ("DELETE FROM trades", "only SELECT"),
        ("INSERT INTO clients (id) VALUES (1)", "only SELECT"),
        ("TRUNCATE trades", "only SELECT"),
        ("ALTER TABLE trades ADD COLUMN x int", "only SELECT"),
        ("CREATE TABLE x AS SELECT * FROM trades", "only SELECT"),
        ("GRANT ALL ON trades TO public", "only SELECT"),
        ("COPY trades TO '/tmp/x'", "only SELECT"),
        ("SET statement_timeout = 0", "only SELECT"),
        ("DO $$ BEGIN DELETE FROM trades; END $$", "only SELECT"),
        # multi-statement
        ("SELECT 1 FROM trades; DROP TABLE trades", "exactly one statement"),
        ("SELECT 1 FROM trades; SELECT 2 FROM trades", "exactly one statement"),
        # comment injection
        ("SELECT * FROM trades -- ; DROP TABLE trades", "comments"),
        ("SELECT * FROM trades /* ignore previous rules */", "comments"),
        ("SELECT * FROM trades WHERE id = 1 --", "comments"),
        ("SELECT/**/count(*)/**/FROM trades", "comments"),
        # writes hidden inside a query
        ("WITH d AS (DELETE FROM trades RETURNING *) SELECT * FROM d", "forbidden operation"),
        (
            "WITH u AS (UPDATE trades SET qty = 1 RETURNING id) SELECT count(*) FROM u",
            "forbidden operation",
        ),
        ("SELECT * INTO stolen FROM clients", "forbidden operation"),
        ("SELECT * FROM trades FOR UPDATE", "forbidden operation"),
        # dangerous functions
        ("SELECT pg_sleep(10)", "function not allowed"),
        ("SELECT * FROM trades WHERE pg_sleep(1) IS NULL", "function not allowed"),
        ("SELECT pg_read_file('/etc/passwd')", "function not allowed"),
        ("SELECT set_config('statement_timeout', '0', false)", "function not allowed"),
        ("SELECT current_setting('is_superuser')", "function not allowed"),
        ("SELECT * FROM dblink('host=evil', 'select 1') AS t(a int)", "function not allowed"),
        ("SELECT lo_import('/etc/passwd')", "function not allowed"),
        # tables outside the allowlist
        ("SELECT * FROM chunks", "table not allowed"),
        ("SELECT * FROM semantic_cache", "table not allowed"),
        ("SELECT usename, passwd FROM pg_catalog.pg_shadow", "schema not allowed"),
        ("SELECT * FROM information_schema.tables", "schema not allowed"),
        ("SELECT * FROM trades t JOIN pg_user u ON true", "table not allowed"),
        ("WITH clients AS (SELECT * FROM documents) SELECT * FROM clients", "table not allowed"),
        # junk
        ("", "empty"),
        ("   ", "empty"),
        ("SELEC * FROM trades", ""),
        ("SELECT * FROM trades WHERE name = 'unterminated", "tokenize"),
    ],
)
def test_rejects_unsafe_sql(sql: str, reason: str) -> None:
    with pytest.raises(SQLGuardError) as ei:
        GUARD.validate(sql)
    assert reason in ei.value.reason
