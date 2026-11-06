FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 UV_COMPILE_BYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv==0.4.30

WORKDIR /app
# CPU-only torch keeps the image ~1.5GB smaller than the default CUDA wheel.
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install ".[rerank]"
COPY alembic.ini ./
COPY migrations ./migrations
COPY scripts ./scripts

RUN useradd --create-home app && mkdir -p /models && chown app /models
USER app
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn enterprise_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1"]
