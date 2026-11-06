"""RAG schema: pgvector extension, documents, chunks (HNSW + GIN), semantic cache.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-06
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBED_DIM = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("source_path", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("doc_type", sa.String(32), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.String(128),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=False),
        # Generated column: Postgres keeps the tsvector in sync with `content`; no triggers.
        sa.Column(
            "tsv",
            TSVECTOR,
            sa.Computed("to_tsvector('english', content)", persisted=True),
        ),
    )
    op.create_index(
        "uq_chunks_document_chunk", "chunks", ["document_id", "chunk_index"], unique=True
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_chunks_tsv_gin", "chunks", ["tsv"], postgresql_using="gin")

    op.create_table(
        "semantic_cache",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("query_text", sa.Text, nullable=False),
        sa.Column("query_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=False),
        sa.Column("response", JSONB, nullable=False),
        sa.Column("hit_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_semantic_cache_embedding_hnsw",
        "semantic_cache",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_semantic_cache_last_accessed", "semantic_cache", ["last_accessed_at"])


def downgrade() -> None:
    op.drop_table("semantic_cache")
    op.drop_table("chunks")
    op.drop_table("documents")
