"""Finance warehouse schema: instruments, clients, accounts, trades.

Revision ID: 0001
Revises:
Create Date: 2026-11-06
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "instruments",
        sa.Column("ticker", sa.String(12), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("asset_class", sa.String(32), nullable=False),
        sa.Column("sector", sa.String(64), nullable=False),
    )
    op.create_table(
        "clients",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("region", sa.String(32), nullable=False),
        sa.Column("segment", sa.String(32), nullable=False),
        sa.Column("onboarded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_clients_region_segment", "clients", ["region", "segment"])

    op.create_table(
        "accounts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "client_id",
            sa.Integer,
            sa.ForeignKey("clients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
    )
    # FK columns are not auto-indexed in Postgres; joins clients->accounts need this.
    op.create_index("ix_accounts_client_id", "accounts", ["client_id"])

    op.create_table(
        "trades",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer,
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ticker", sa.String(12), sa.ForeignKey("instruments.ticker"), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("qty", sa.Integer, nullable=False),
        sa.Column("price", sa.Numeric(14, 4), nullable=False),
        sa.Column("traded_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("side IN ('BUY', 'SELL')", name="ck_trades_side"),
        sa.CheckConstraint("qty > 0", name="ck_trades_qty_positive"),
        sa.CheckConstraint("price > 0", name="ck_trades_price_positive"),
    )
    # Composite indexes chosen for the common access paths:
    #  * per-account history ordered by time
    #  * per-instrument activity over a date range
    #  * global date-range scans / monthly aggregates
    op.create_index("ix_trades_account_id_traded_at", "trades", ["account_id", "traded_at"])
    op.create_index("ix_trades_ticker_traded_at", "trades", ["ticker", "traded_at"])
    op.create_index("ix_trades_traded_at", "trades", ["traded_at"])


def downgrade() -> None:
    op.drop_table("trades")
    op.drop_table("accounts")
    op.drop_table("clients")
    op.drop_table("instruments")
