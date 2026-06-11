# Regulatory Intelligence System

Multi-agent RAG over **live SEC EDGAR filings** + synthetic internal compliance docs.
Phase 0+1: foundations + hybrid retrieval spine.

## Architecture
- **LLM layer**: pluggable provider (Ollama Cloud default, Claude optional), role-tiered.
- **Embeddings**: bge-m3 dense (via Ollama) + FastEmbed BM25 sparse.
- **Store**: Qdrant `corpus` collection, dense+sparse named vectors, server-side RRF fusion + payload filters.
- **Retrieval**: hybrid search -> LLM listwise rerank (`RetrieverAgent`).

## Setup
1. `uv sync`
2. `cp .env.example .env` and set `SEC_USER_AGENT` to "Your Name your@email".
3. Pull the embedding model: `ollama pull bge-m3` (gpt-oss:120b-cloud already available).
4. Start Qdrant: `docker compose up -d`  (or set `QDRANT_EMBEDDED=true` to skip Docker).

## Demo
```bash
uv run python -m regintel.cli ingest --sec-query "insider trading policy" --sec-limit 5
uv run python -m regintel.cli query "What are our obligations around insider trading?" --jurisdiction US-SEC
```

## Tests
- `uv run pytest`            # fast unit/integration (mocked)
- `uv run pytest -m live`    # requires Ollama + (embedded) Qdrant
