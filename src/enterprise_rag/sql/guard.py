"""SQLGuard: static validation of LLM-generated SQL before it reaches the database.

The guard is an allowlist, not a blocklist: a statement passes only if it is exactly one
read-only query over the approved tables. It is the first of three layers – the executor
also runs inside a READ ONLY transaction with a statement timeout, as a Postgres role that
only has SELECT on the finance tables.

Checks, in order:
1. no SQL comments anywhere (comments are how ``SELECT 1 -- '; DROP ...`` style payloads
   and prompt-injected instructions hide),
2. parses as exactly one statement (no ``;``-chained statements),
3. the root is a query (SELECT / UNION / INTERSECT / EXCEPT, optionally with CTEs),
4. no write / DDL / session nodes anywhere in the tree (INSERT, UPDATE, DELETE, DROP,
   CREATE, ALTER, MERGE, TRUNCATE, COPY, SET, SELECT ... INTO, FOR UPDATE locks, ...),
5. no dangerous functions (``pg_sleep``, file access, ``dblink``, ``set_config``, ...),
6. every table referenced is an allowed table or a CTE defined in the query,
7. a LIMIT is enforced: added if missing, clamped if larger than ``max_rows``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Merge,
    exp.TruncateTable,
    exp.Copy,
    exp.Set,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Into,
    exp.Lock,
    exp.Grant,
    exp.Use,
    exp.Pragma,
    exp.LoadData,
)

FORBIDDEN_FUNCTIONS = frozenset(
    {
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_advisory_lock",
        "pg_advisory_xact_lock",
        "lo_import",
        "lo_export",
        "lo_get",
        "dblink",
        "dblink_exec",
        "set_config",
        "current_setting",
        "query_to_xml",
        "txid_current",
    }
)
FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink")


class SQLGuardError(ValueError):
    """The statement was rejected; ``reason`` is safe to show the LLM for a repair."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class GuardedSQL:
    sql: str  # normalised, LIMIT-enforced SQL that will actually be executed
    tables: frozenset[str]
    limit_applied: bool


class SQLGuard:
    def __init__(
        self,
        allowed_tables: Iterable[str],
        *,
        max_rows: int = 1000,
        dialect: str = "postgres",
    ) -> None:
        self.allowed_tables = frozenset(t.lower() for t in allowed_tables)
        self.max_rows = max_rows
        self.dialect = dialect

    def validate(self, sql: str) -> GuardedSQL:
        sql = sql.strip()
        if not sql:
            raise SQLGuardError("empty statement")
        self._reject_comments(sql)
        tree = self._parse_single(sql)

        if not isinstance(tree, exp.Query):
            raise SQLGuardError(f"only SELECT queries are allowed, got {tree.key.upper()}")
        for node in tree.walk():
            if isinstance(node, FORBIDDEN_NODES):
                raise SQLGuardError(f"forbidden operation: {node.key.upper()}")
            if isinstance(node, exp.Func):
                self._check_function(node)

        tables = self._check_tables(tree)
        limited_tree, limited = self._enforce_limit(tree)
        return GuardedSQL(limited_tree.sql(dialect=self.dialect), tables, limited)

    # ---------------------------------------------------------------------- checks ----
    def _reject_comments(self, sql: str) -> None:
        try:
            tokens = sqlglot.tokenize(sql, read=self.dialect)
        except TokenError as e:
            raise SQLGuardError(f"could not tokenize SQL: {e}") from e
        if any(t.comments for t in tokens):
            raise SQLGuardError("SQL comments are not allowed")

    def _parse_single(self, sql: str) -> exp.Expression:
        try:
            statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        except ParseError as e:
            raise SQLGuardError(f"could not parse SQL: {str(e).splitlines()[0]}") from e
        if len(statements) != 1:
            raise SQLGuardError(f"exactly one statement allowed, got {len(statements)}")
        statement = statements[0]
        if not isinstance(statement, exp.Expression):
            raise SQLGuardError("unsupported statement")
        return statement

    def _check_function(self, node: exp.Func) -> None:
        name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
        if name in FORBIDDEN_FUNCTIONS or name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
            raise SQLGuardError(f"function not allowed: {name}")

    def _check_tables(self, tree: exp.Expression) -> frozenset[str]:
        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        used: set[str] = set()
        for table in tree.find_all(exp.Table):
            name = table.name.lower()
            if not name:  # e.g. table-valued function such as generate_series(...)
                continue
            schema = (table.db or "").lower()
            if schema and schema != "public":
                raise SQLGuardError(f"schema not allowed: {schema}")
            if name in cte_names and not schema:
                continue
            if name not in self.allowed_tables:
                raise SQLGuardError(f"table not allowed: {name}")
            used.add(name)
        return frozenset(used)

    def _enforce_limit(self, tree: exp.Query) -> tuple[exp.Query, bool]:
        if isinstance(tree, exp.Select):
            current = tree.args.get("limit")
            value = _literal_int(current.expression) if current is not None else None
            if value is not None and value <= self.max_rows:
                return tree, False
            return tree.limit(self.max_rows, copy=True), True
        # UNION / INTERSECT / EXCEPT: wrap so the limit applies to the combined result.
        wrapped = exp.select("*").from_(tree.subquery("guarded")).limit(self.max_rows)
        return wrapped, True


def _literal_int(node: exp.Expression | None) -> int | None:
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return int(node.this)
        except ValueError:
            return None
    return None
