PY ?= .venv/bin/python
COMPOSE ?= docker compose

.PHONY: ui install up up-ollama down migrate seed docs ingest eval bench explain test test-unit lint typecheck check health api

install:
	uv venv --python 3.11 .venv && uv pip install --python .venv -e ".[dev,rerank,ui]"

up:            ## start postgres + redis + api + ui (uses host Ollama by default)
	$(COMPOSE) up -d --build postgres redis api ui

up-ollama:     ## also run Ollama in a container and pull the models
	$(COMPOSE) --profile ollama up -d ollama
	$(COMPOSE) exec ollama ollama pull llama3.1:8b
	$(COMPOSE) exec ollama ollama pull nomic-embed-text
	OLLAMA_URL=http://ollama:11434 $(COMPOSE) up -d --build postgres redis api ui

down:
	$(COMPOSE) down

migrate:
	$(PY) -m alembic upgrade head

seed: migrate
	$(PY) -m scripts.seed_db --trades 100000

docs:          ## regenerate the synthetic document corpus
	$(PY) -m scripts.generate_docs

ingest:
	$(PY) -m scripts.ingest

eval:
	$(PY) -m scripts.eval

explain:
	$(PY) -m scripts.explain

bench:
	$(PY) -m scripts.bench

api:
	$(PY) -m uvicorn enterprise_rag.api.app:create_app --factory --port 8000

ui:            ## Streamlit front-end (expects the API on RAG_API_URL, default :8000)
	$(PY) -m streamlit run ui/streamlit_app.py --server.port 8501

lint:
	$(PY) -m ruff check src tests scripts migrations ui
	$(PY) -m ruff format --check src tests scripts migrations ui

typecheck:
	$(PY) -m mypy

test:
	$(PY) -m pytest --cov --cov-report=term-missing

test-unit:
	$(PY) -m pytest tests/unit

check: lint typecheck test

health:
	./scripts/healthcheck.sh
