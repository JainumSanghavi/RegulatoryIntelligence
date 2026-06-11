# Regulatory Intelligence System — Phase 0+1 Design (Foundations + Retrieval Spine)

**Date:** 2026-06-11
**Status:** Approved (brainstorming) — pending spec review
**Scope of this spec:** Phase 0 (Foundations) + Phase 1 (Ingestion + Retrieval spine). Later phases are summarized for context only and will each get their own spec.

---

## 1. Project context

A **portfolio project** by an ML/AI engineer (on F1, seeking US full-time) to **demonstrate multi-agent orchestration complexity**. Success = an impressive, runnable, well-documented demonstration — not a production deployment.

The full system is a multi-agent RAG pipeline that monitors regulatory feeds, ingests internal company documents, and answers compliance questions with cited, grounded reports. It is decomposed into 6 phases; this spec covers the first two.

### Phase roadmap (context)

| Phase | Name | Result |
|---|---|---|
| **0** | Foundations | Scaffold, config, pluggable LLM layer, typed state, Qdrant + schema |
| **1** | Ingestion + Retrieval spine | SEC EDGAR live ingest → Qdrant → hybrid search + LLM rerank (RetrieverAgent) |
| 2 | Orchestration + reasoning | LangGraph StateGraph; Orchestrator, Analyst, ImpactAssessor, Reporter |
| 3 | Evaluation | EvaluatorAgent: RAGAS + LLM-as-judge faithfulness/citation/conflict |
| 4 | Monitor + scheduling | APScheduler SEC polling, diff detection, regulation_changelog |
| 5 | API + UI + polish | FastAPI + minimal web UI, demo script, docs |

**Build order:** 0 → 1 → 2 → 3 → 4 → 5.

---

## 2. Key decisions (locked)

- **Build vs runtime split:** All build/coding done by Claude. The *product's runtime* agents call **Ollama Cloud** models (Ollama Desktop at `http://localhost:11434`, connected to ollama.com) to save tokens.
- **Build-time Claude tiering:** Opus = planning + tricky integration; Sonnet = bulk implementation; Haiku = mechanical work. (Applied during implementation, not this spec.)
- **Tiered production model routing (pluggable default):**
  - Analyst, Reporter, Orchestrator, **LLM reranker** → `gpt-oss:120b` (Ollama)
  - ImpactAssessor, Evaluator → frontier (`kimi-k2.6` / `deepseek-v4-pro`, Ollama)
  - Claude available as a per-role drop-in via config/env.
- **Live source:** **SEC EDGAR** only (`data.sec.gov`, no account/key, requires `User-Agent` header). Other sources mocked in later phases.
- **Demo surface:** FastAPI + minimal web UI (Phase 5).
- **Internal docs:** small **synthetic, self-generated** Markdown corpus (committed, KB-sized), tuned to have deliberate gaps/overlaps with SEC content.
- **Embeddings:** **bge-m3 via Ollama** (local pull, ~1.2 GB) — **dense vectors only** (Ollama does not expose bge-m3's sparse output).
- **Sparse/lexical:** **FastEmbed BM25** (non-neural tokenizer + IDF, kilobytes) → Qdrant sparse vectors. Together with dense = the "hybrid dense + BM25" requirement.
- **Reranker:** **LLM-as-reranker via Ollama Cloud** (`gpt-oss:120b` listwise), 0 local disk.
- **Constraints:** minimize local disk; avoid any service requiring account signup.

### Honest caveat captured
bge-m3's dense+sparse+multivector outputs are only available via the `FlagEmbedding` library (~2.2 GB download). Via Ollama we get **dense only**; sparse comes from FastEmbed BM25. Same hybrid capability, low disk.

---

## 3. Toolchain (verified present 2026-06-11)

- Python 3.12 (anaconda) ✅
- Docker 28.4 ✅ (daemon running)
- Ollama 0.24 ✅; `gpt-oss:120b-cloud` + `bge-m3:latest` (1.2 GB) pulled ✅
- `uv` 0.11.21 ✅ (package/venv manager)
- Dependencies declared in `pyproject.toml`, installed via `uv sync`. User installs nothing else.

---

## 4. Module layout (`src/regintel/`)

```
config.py            # pydantic-settings: model routing table, Qdrant URL, SEC user-agent, ollama host
state.py             # typed AgentState (TypedDict) — FULL LangGraph schema defined now
llm/
  base.py            # LLMProvider protocol: chat(), chat_structured(messages, schema)
  ollama_provider.py # Ollama via native /api/chat (+ JSON-schema `format` for structured)
  claude_provider.py # Anthropic SDK adapter (same interface)
  router.py          # role -> (provider, model) resolution from config
embeddings/
  ollama_embedder.py # bge-m3 dense via Ollama /api/embed
  sparse.py          # FastEmbed BM25 sparse encoder
store/
  schema.py          # payload metadata schema + collection definitions
  qdrant_store.py    # collection mgmt, idempotent upsert, hybrid Query-API search + RRF
ingest/
  sec_edgar.py       # live SEC fetcher (+ on-disk cache)
  internal_docs.py   # synthetic corpus loader
  chunker.py         # RecursiveCharacterTextSplitter wrapper
  pipeline.py        # fetch -> clean -> chunk -> embed -> upsert
rerank/
  llm_reranker.py    # listwise rerank via Ollama
agents/
  retriever.py       # RetrieverAgent
data/
  internal/          # synthetic docs (committed, tiny)
  raw/ cache/        # gitignored
tests/
docker-compose.yml   # Qdrant
.env.example
```

---

## 5. Phase 0 — Foundations

### 5.1 Pluggable LLM layer
`LLMProvider` protocol with:
- `chat(messages, *, model, temperature, **opts) -> str`
- `chat_structured(messages, *, model, schema) -> dict` — validated structured output

Implementations: `OllamaProvider` (native `/api/chat`; structured via the `format` JSON-schema field), `ClaudeProvider` (Anthropic SDK; structured via tool-use/response format). `router.resolve(role) -> (provider, model)` reads a config table; env vars override per role. Defaults per the tiered routing above.

Structured-output reliability: schema passed to the model; on parse/validation failure, one re-ask with a schema reminder, then graceful fallback with a logged error.

### 5.2 Config (`config.py`)
`pydantic-settings` `Settings` from `.env`:
- `OLLAMA_HOST` (default `http://localhost:11434`)
- `ANTHROPIC_API_KEY` (optional; only if a Claude role is enabled)
- `SEC_USER_AGENT` (required for ingestion, e.g. `"Jainum Sanghavi sanghavi.j@northeastern.edu"`)
- `QDRANT_URL` (default `http://localhost:6333`) + `QDRANT_EMBEDDED` fallback flag
- Model-routing overrides per role.

### 5.3 Typed `AgentState` (`state.py`)
Full LangGraph `TypedDict` defined now (Phases 0+1 populate only the retrieval fields):
```python
class AgentState(TypedDict, total=False):
    query: str
    sub_questions: list[str]
    filters: RetrievalFilters          # jurisdiction, doc_type, source, date range
    retrieved: list[RetrievedChunk]    # ranked chunks + scores + metadata
    analyst_findings: list[Finding]    # Phase 2
    impact_assessments: list[Impact]   # Phase 2
    report: Report | None              # Phase 2
    eval_scores: EvalScores | None     # Phase 3
    errors: list[str]
    messages: list[dict]               # message-passing between agents
```

### 5.4 Vector store + schema (`store/`)
**Shared collection** `corpus` (SEC regs + internal docs together; honors the "shared collection, metadata filters" decision). Named vectors: `dense` (1024-dim, cosine) + `sparse` (BM25). `regulation_changelog` collection deferred to Phase 4.

Payload / metadata schema:
```python
{
  "doc_id": str, "chunk_index": int, "text": str,
  "source": "sec" | "internal",
  "jurisdiction": "US-SEC" | "internal",
  "doc_type": "filing" | "policy" | "sop" | "contract",
  "title": str, "url": str | None,
  "regulation_id": str | None,
  "form_type": str | None,      # SEC: 10-K, 8-K, ...
  "accession_no": str | None,   # SEC
  "effective_date": str | None, # ISO
  "filed_date": str | None,     # ISO
}
```
Qdrant runs via `docker-compose`; embedded `:memory:`/on-disk fallback if the daemon is down. Startup health-check.

---

## 6. Phase 1 — Ingestion + Retrieval spine

### 6.1 SEC EDGAR ingestion (`ingest/sec_edgar.py`)
- Live fetch from `data.sec.gov` / `efts.sec.gov` with required `User-Agent` header.
- Two retrieval modes (configurable): EDGAR full-text search (`efts.sec.gov/LATEST/search-index?q=...&forms=...`), or company filing history (`data.sec.gov/submissions/CIK##########.json`) for a small configurable set of CIKs/forms (e.g. recent 8-K / 10-K risk & compliance sections).
- Download primary document (HTML) → clean text (`selectolax`/`BeautifulSoup`).
- **On-disk cache** under `data/cache/` (gitignored): live fetch, but respect SEC's ~10 req/s limit and keep demos reproducible. Throttle to stay under the limit.

### 6.2 Synthetic internal docs (`ingest/internal_docs.py`)
Loads the small committed Markdown corpus (`data/internal/`). Authored to contain deliberate gaps/overlaps vs SEC content (e.g. an insider-trading policy missing a blackout-window clause). Same chunk→embed→upsert path, tagged `source=internal`.

### 6.3 Chunking (`ingest/chunker.py`)
LangChain `RecursiveCharacterTextSplitter`, ~800-token chunks, ~150 overlap, token-aware (`tiktoken`). Preserves `chunk_index` + parent `doc_id`.

### 6.4 Indexing (`store/qdrant_store.py`)
Per chunk: dense (bge-m3/Ollama) + sparse (FastEmbed BM25) → upsert to `corpus` with payload. **Idempotent**: deterministic point IDs from `doc_id`+`chunk_index` so re-ingest updates, not duplicates.

### 6.5 RetrieverAgent (`agents/retriever.py`) — Phase 1 headliner
1. Accept query + filters (`jurisdiction`, `doc_type`, `source`, date range).
2. Embed query (dense) + encode (sparse).
3. **Qdrant Query API**: prefetch dense top-50 + sparse top-50, **RRF** fusion server-side → top-20, with payload filters applied.
4. **LLM listwise rerank** (Ollama `gpt-oss:120b`): feed 20 candidates → reranked top-k (≈8) with brief relevance rationale (structured JSON).
5. Return ranked chunks + scores + metadata in the shape consumed by `AgentState.retrieved`.

### 6.6 Error handling
- Ollama unreachable → retry w/ backoff + clear message.
- SEC rate-limit/5xx → backoff + serve from cache.
- Structured-output parse failure → one re-ask w/ schema reminder → graceful fallback.
- Qdrant down → embedded fallback; startup health-check.

---

## 7. Testing strategy (TDD)

- **Unit:** chunker; payload builder; BM25 encoder; point-ID determinism; router role→model resolution; RRF fusion ordering.
- **Integration:** embedded Qdrant (`:memory:`) + seeded docs → assert metadata filtering + fused ranking; full ingest→retrieve on a tiny fixture set.
- **SEC fetcher:** recorded real response replayed as a fixture (no live hits in CI).
- **Providers:** mock the Ollama HTTP layer; a live smoke test behind `@pytest.mark.live` (requires Ollama running).
- TDD discipline: tests authored before/with implementation per task.

---

## 8. Definition of done (Phase 0+1)

1. `uv sync` installs cleanly; `pytest` green (non-live).
2. `provider.chat()` returns text from `gpt-oss:120b`; `chat_structured()` returns schema-valid JSON.
3. `docker-compose up` brings Qdrant up; collection `corpus` created with dense+sparse vectors.
4. Ingest command pulls real SEC filings + synthetic internal docs into `corpus` (idempotent).
5. RetrieverAgent answers "ask a question → ranked, jurisdiction-filtered, LLM-reranked chunks with metadata."
6. README documents setup + a one-command retrieval demo.

---

## 9. Out of scope (this spec)

LangGraph orchestration, Analyst/ImpactAssessor/Reporter/Evaluator agents, RAGAS eval, MonitorAgent/scheduling, FastAPI/UI. Each is a later phase with its own spec.
