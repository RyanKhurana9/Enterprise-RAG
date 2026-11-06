"""Index chunks.content_hash so re-ingestion can reuse embeddings of unchanged chunks.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-06
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_chunks_content_hash", "chunks", ["content_hash"])


def downgrade() -> None:
    op.drop_index("ix_chunks_content_hash", table_name="chunks")
