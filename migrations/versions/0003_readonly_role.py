"""Read-only Postgres role for the Text2SQL executor.

Defence in depth: even if SQLGuard were bypassed, this role can only SELECT from the four
finance tables, every transaction is read-only, and statements time out after 5s.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-06
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ROLE = "rag_readonly"
FINANCE_TABLES = ("clients", "accounts", "trades", "instruments")


def upgrade() -> None:
    password = os.environ.get("RAG_READONLY_PASSWORD", "readonly").replace("'", "''")
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{ROLE}') THEN
                CREATE ROLE {ROLE} LOGIN PASSWORD '{password}';
            ELSE
                ALTER ROLE {ROLE} LOGIN PASSWORD '{password}';
            END IF;
        END
        $$;
        """
    )
    op.execute(f"ALTER ROLE {ROLE} SET default_transaction_read_only = on")
    op.execute(f"ALTER ROLE {ROLE} SET statement_timeout = '5s'")
    op.execute(f"ALTER ROLE {ROLE} SET idle_in_transaction_session_timeout = '10s'")
    op.execute(f"ALTER ROLE {ROLE} CONNECTION LIMIT 20")
    op.execute(f'GRANT CONNECT ON DATABASE "{_dbname()}" TO {ROLE}')
    op.execute(f"GRANT USAGE ON SCHEMA public TO {ROLE}")
    op.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {ROLE}")
    op.execute(f"GRANT SELECT ON {', '.join(FINANCE_TABLES)} TO {ROLE}")


def _dbname() -> str:
    bind = op.get_bind()
    return str(bind.exec_driver_sql("SELECT current_database()").scalar_one())


def downgrade() -> None:
    op.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {ROLE}")
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {ROLE}")
    op.execute(f'REVOKE CONNECT ON DATABASE "{_dbname()}" FROM {ROLE}')
    op.execute(f"DROP ROLE IF EXISTS {ROLE}")
