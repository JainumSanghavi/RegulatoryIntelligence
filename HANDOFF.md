# RegIntel — Handoff / Working Notes

A practical orientation for continuing this project in a fresh chat (e.g. with Sonnet). Your stated next goals: **(1) expand the data, (2) understand the app, (3) find more demo examples.** This doc is organized around those.

---

## 0. TL;DR — what this is and where it stands

**Regulatory Intelligence System** — a multi-agent RAG app that answers compliance questions over **live SEC EDGAR filings + internal company policies**, with grounded citations and an automatic faithfulness check. Built as a portfolio piece demonstrating multi-agent orchestration.

- **Status: all 6 phases complete (0–5).** Runs end-to-end via CLI or a web UI. 105 automated tests pass; ruff clean. Everything committed and pushed to `main` (GitHub: `JainumSanghavi/RegulatoryIntelligence`).
- **LLMs run on Ollama** (local desktop → Ollama Cloud). No paid API in the runtime. Embeddings = `bge-m3` (local, ~1.2 GB). Chat = `gpt-oss:120b-cloud`; the hard-reasoning agents (ImpactAssessor, Evaluator) use a frontier model `kimi-k2.6:cloud`.

---

## 1. Run it (quick reference)

```bash
uv sync                                    # install deps (creates .venv)
cp .env.example .env                       # set SEC_USER_AGENT to "Your Name your@email"
ollama pull bge-m3                          # embedding model (chat models are cloud)

# Use embedded Qdrant (no Docker) for everything by exporting:
export QDRANT_EMBEDDED=true
export SEC_USER_AGENT="Your Name your@email.com"

# 1. Ingest data (internal docs + live SEC filings) into ./qdrant_storage
uv run python -m regintel.cli ingest --sec-query "insider trading policy" --sec-limit 3

# 2. Ask (full multi-agent pipeline; ~30-90s on real models)
uv run python -m regintel.cli ask "Does our insider trading policy comply with SEC blackout window requirements?"

# 3. Monitor (detect new SEC filings -> auto-ingest -> changelog)
uv run python -m regintel.cli monitor --once --query "insider trading policy" --forms 8-K --limit 3
uv run python -m regintel.cli changelog

# 4. Web UI -> open http://localhost:8000
uv run python -m regintel.cli serve
#   stop it later:  pkill -f "regintel.cli serve"

# Also useful:
uv run python -m regintel.cli query "insider trading" --jurisdiction US-SEC   # raw retrieval, no LLM reasoning
uv run pytest -q                  # fast tests (mocked, no models)
uv run pytest -m live -q          # live tests (need Ollama running + network)
```

**Qdrant modes:** `QDRANT_EMBEDDED=true` → on-disk `./qdrant_storage` (no Docker, single-process). Unset / `false` → talks to a Qdrant server at `QDRANT_URL` (`docker compose up -d`). The web `serve` + `ask` + `monitor` all read the same `./qdrant_storage` in embedded mode, so **ingest first, then serve** so the UI has data.

---

## 2. Understand the app (architecture map)

Full prose explanation is in **`README.md`** (problem framing + deep dive). The blow-by-blow design + implementation reasoning for each phase is in **`docs/superpowers/specs/`** and **`docs/superpowers/plans/`** (5 spec/plan pairs, dated). Read those to understand *why* things are shaped as they are.

**Request flow (query time)** — a LangGraph state machine:
```
ask "question"
  -> Orchestrator.classify()  -> LOOKUP | GAP_CHECK | IMPACT     (agents/orchestrator.py)
  -> Retriever (hybrid dense+sparse + RRF + LLM rerank)          (agents/retriever.py)
  -> [GAP_CHECK/IMPACT only] retrieve internal docs too
  -> Analyst: extract clauses, find gaps                          (agents/analyst.py)
  -> [if gaps] ImpactAssessor: affected policies + severity       (agents/impact_assessor.py)  [frontier model]
  -> Reporter: assemble cited answer                              (agents/reporter.py)
  -> Evaluator: faithfulness / citation-coverage / conflicts      (agents/evaluator.py)        [frontier model]
  -> Report (answer + citations + findings + impacts + eval)
```
The graph wiring + routing live in **`src/regintel/orchestration/graph.py`** and **`nodes.py`**. The entry point is `run_query(query, *, graph)`.

**Monitoring (decoupled)** — `src/regintel/monitoring/agent.py` (`MonitorAgent.poll()`) + `scheduler.py` (APScheduler). It detects unseen SEC filings, auto-ingests them into the corpus, and records each in the `regulation_changelog` collection.

**Key building blocks:**
| Concern | File |
|---|---|
| Config (models, Qdrant, SEC UA) | `src/regintel/config.py` |
| Typed data contracts (Report, Citation, Finding, Impact, EvalScores, QueryType, ChangelogEntry) | `src/regintel/types.py` |
| LangGraph state | `src/regintel/state.py` |
| LLM provider layer (Ollama + Claude adapter, role→model router) | `src/regintel/llm/` |
| Embeddings: bge-m3 dense + FastEmbed BM25 sparse | `src/regintel/embeddings/` |
| Qdrant: shared `corpus` collection + hybrid search; `changelog_store.py` | `src/regintel/store/` |
| Ingestion: SEC client, chunker, internal-doc loader, pipeline | `src/regintel/ingest/` |
| FastAPI app + single-page UI | `src/regintel/api/app.py`, `api/static/index.html` |
| CLI (all commands) | `src/regintel/cli.py` |

**The two reasoning data structures to know:**
- `Finding` (Analyst output): `{topic, requirement, internal_status, gap: bool, explanation, citations}`
- `Impact` (Assessor output): `{topic, affected_policies, severity: low|medium|high|critical, rationale}`
- `EvalScores` (Evaluator): `{faithfulness, citation_coverage, conflicts, flagged, notes}` — `flagged=True` if faithfulness < 0.7 or conflicts exist.

---

## 3. Expand the data (your main goal)

Everything is one shared Qdrant collection called **`corpus`**, holding both SEC filings and internal docs, distinguished by payload metadata: `source` (`sec`|`internal`), `jurisdiction` (`US-SEC`|`internal`), `doc_type` (`filing`|`policy`|`sop`|`contract`), plus `title`, `url`, `form_type`, `filed_date`, `accession_no`, `chunk_index`. Re-ingesting the same doc is idempotent (deterministic point IDs).

### 3a. Add more SEC content
The CLI `ingest` uses SEC EDGAR **full-text search** then fetches each filing's real body. Just vary the query/forms:
```bash
uv run python -m regintel.cli ingest --sec-query "records retention rule 17a-4" --forms 10-K --sec-limit 5
uv run python -m regintel.cli ingest --sec-query "disclosure controls and procedures" --sec-limit 5
uv run python -m regintel.cli ingest --sec-query "cybersecurity risk governance" --sec-limit 5
```
- `--forms` accepts a comma list (`8-K,10-K`). `--sec-limit` caps how many filings.
- Mechanism lives in `src/regintel/ingest/sec_edgar.py` (`SECClient.full_text_search` + `fetch_document` + `build_doc_url`) and `sec_ingest.py` (`sec_query_to_docs`). Responses are cached under `data/cache/` (gitignored), rate-limited to be polite to SEC.
- **Important for demos:** a GAP_CHECK only finds good gaps if the corpus has SEC content *on the same topic* as an internal doc. Right now the SEC content is insider-trading-heavy. To make the retention/disclosure internal docs "checkable," ingest matching SEC topics (examples above).

### 3b. Add more internal docs
Drop Markdown files into **`data/internal/`** (they're committed, tiny). The loader (`ingest/internal_docs.py`) infers `doc_type` from the filename: contains `sop`→sop, `contract`/`agreement`→contract, else `policy`. The first `#` heading becomes the title. Then re-run `ingest`.
- The 3 existing files have **deliberate compliance gaps** (e.g. the insider-trading policy is missing a blackout-window clause) so gap analysis surfaces interesting findings. Follow that pattern: write a realistic internal doc that *partially* matches a regulation, leaving a clear gap.

### 3c. Other jurisdictions / sources (bigger lift)
Only SEC is wired for live fetch (by design). Adding EUR-Lex/FDA/GOV.UK would mean a new client like `sec_edgar.py` + a `*_ingest.py` bridge. The retrieval/agents are source-agnostic (they filter by `jurisdiction`/`source` metadata), so the downstream pipeline needs no changes.

---

## 4. Demo examples (your third goal)

The contrast between query types is the most compelling thing to show — see the menu below. Pick a few, run them, and screenshot the rendered reports for your demo.

- **LOOKUP** (fast, no gap analysis): "What does an SEC insider trading policy require around blackout windows?" / "What is a 10b5-1 trading plan?"
- **GAP_CHECK** (the showcase — finds + scores gaps): "Does our insider trading policy comply with SEC blackout window requirements?" / "Does our insider trading policy address 10b5-1 plans the way the SEC expects?"
- **IMPACT** (severity-weighted): "A new SEC rule tightens quarterly blackout windows — how does that affect our policies?"

**For your demo, highlight:** (a) the Orchestrator routing to different paths, (b) gap findings with **severity badges**, (c) citations linking to **real SEC filings**, (d) the **Evaluation strip** flagging low-faithfulness answers — that self-check is the differentiator vs a plain chatbot. To make a *new* example shine, first ingest SEC content on that topic (§3a), then write/keep an internal doc with a deliberate gap on it (§3b).

---

## 5. Conventions & non-obvious gotchas (read before editing)

- **Package manager:** `uv`. Run everything as `uv run ...`. Dev deps are in `[dependency-groups]`, so plain `uv run pytest` works (do NOT use `--extra dev`).
- **Git:** commits authored as `Jainum Sanghavi <sanghavi.h.j20@gmail.com>`; **no "Co-Authored-By: Claude" trailer** (the owner's preference). Conventional-commit messages (`feat:`/`fix:`/`docs:`/`chore:`).
- **Tests:** live tests are gated behind `@pytest.mark.live` and deselected by default; run with `-m live` (needs Ollama running). Unit tests mock the LLM/HTTP, so they're fast and offline.
- **Ollama Cloud quirk (already handled, don't undo):** cloud models do NOT enforce the JSON-schema `format` param. `OllamaProvider.chat_structured` works around it by embedding the schema in the prompt, parsing robustly, capping output with `num_predict=8192`, and reprompting once on a parse miss. If you add a new structured LLM call, go through `chat_structured`.
- **Frontier model tags need `:cloud`** (e.g. `kimi-k2.6:cloud`, not `kimi-k2.6`). Configurable via `OLLAMA_FRONTIER_MODEL`. If it isn't pulled, the ImpactAssessor/Evaluator degrade gracefully (skip + warning) rather than crash.
- **Embedded Qdrant = one client per process.** The API shares a single `QdrantClient` across the graph + changelog for this reason (see `api/app.py`). Don't open a second client on the same path.
- **Model routing** is per-agent-role in `src/regintel/llm/router.py` (chat tier vs frontier tier). The whole LLM layer is provider-pluggable — you could point a role at Claude via config without code changes.

---

## 6. Suggested next steps (menu)

- **Broaden the corpus** so all three internal docs have matching SEC regulations (run the §3a ingests for retention + disclosure).
- **Add 1–2 more internal docs** with deliberate gaps (e.g. a cybersecurity-disclosure policy) to widen the demo.
- **Curate a demo script**: 3–4 questions (one per query type) + screenshots of the rendered reports.
- **(Optional) README screenshots / a short architecture write-up** for the CV.
- **(Optional, out of scope so far)** auth, deployment config, RAGAS as an optional extra, content-level version diffing of filings.

Design rationale for every decision is in `docs/superpowers/specs/` and `docs/superpowers/plans/`. When in doubt, read the spec for that phase.
