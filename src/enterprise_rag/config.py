"""Central configuration. Every threshold lives here and can be overridden via env vars.

Env vars use the ``RAG_`` prefix, e.g. ``RAG_SEMANTIC_CACHE_THRESHOLD=0.95``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- Postgres -------------------------------------------------------------------------
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    # Read-only role used *only* by the SQL executor (Text2SQL path).
    readonly_database_url: str = "postgresql+asyncpg://rag_readonly:readonly@localhost:5432/rag"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout_s: float = 10.0

    # --- Redis ----------------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- Ollama ---------------------------------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    embed_batch_size: int = 32
    llm_timeout_s: float = 120.0
    llm_temperature: float = 0.0
    llm_num_ctx: int = 8192

    # --- Chunking -------------------------------------------------------------------------
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    docs_dir: str = "data/docs"

    # --- Retrieval ------------------------------------------------------------------------
    retrieval_candidates: int = 20  # top-N fed into the reranker
    final_top_k: int = 5  # top-K given to the LLM
    rrf_k: int = 60
    hnsw_ef_search: int = 64
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Below this cross-encoder logit for the best chunk we refuse without calling the LLM.
    min_rerank_score: float = -6.0

    # --- SQL ------------------------------------------------------------------------------
    sql_max_rows: int = 1000
    sql_statement_timeout_ms: int = 5000
    sql_allowed_tables: tuple[str, ...] = ("clients", "accounts", "trades", "instruments")
    sql_repair_attempts: int = 1

    # --- Routing --------------------------------------------------------------------------
    router_use_llm: bool = True

    # --- Cache ----------------------------------------------------------------------------
    cache_enabled: bool = True
    exact_cache_ttl_s: int = 3600
    semantic_cache_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    semantic_cache_max_entries: int = 10_000
    semantic_cache_ttl_s: int = 24 * 3600


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
