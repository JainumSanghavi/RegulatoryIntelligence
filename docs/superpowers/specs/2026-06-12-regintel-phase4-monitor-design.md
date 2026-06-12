# Regulatory Intelligence System — Phase 4 Design (Monitor / Scheduled Change Detection)

**Date:** 2026-06-12
**Status:** Approved (brainstorming) — continuous execution authorized
**Scope:** Phase 4 — the MonitorAgent: a scheduled, query-time-decoupled job that polls SEC EDGAR, detects filings not seen before, ingests them into the queryable corpus, and records a changelog entry per change. Builds on the Phase 1 ingestion stack.

---

## 1. Context

Phases 0–3 answer compliance questions at query time. Phase 4 makes the knowledge base **stay current on its own**: a background job watches SEC EDGAR, and when new filings appear it pulls them into the `corpus` (so the query pipeline immediately sees them) and logs what changed. This is the "always-current knowledge" story — the system answers against today's filings, not a frozen snapshot.

---

## 2. Decisions (locked in brainstorming)

- **Detect → ingest → log.** Each poll: find new filings, ingest their bodies into `corpus`, and write a changelog entry. Monitored changes become queryable (closes the loop). No auto impact-assessment (keeps Monitor decoupled from query-time agents).
- **`regulation_changelog` Qdrant collection** is the seen-record AND a searchable log. One point per filing, keyed by `uuid5(accession_no)`; dense vector = embedded summary; payload carries the metadata + summary.
- **Cold start**: first poll records the baseline (all hits new); later polls surface only genuinely new filings.
- **Run modes**: `monitor --once` (single poll, for demo/test) and `monitor --interval N` (long-running APScheduler `BlockingScheduler`). Decoupled — its own process.
- **Resilience**: per-filing failures are skipped; poll errors are logged; the scheduler survives poll failures.

---

## 3. Data contract

Added to `src/regintel/types.py`:
```python
@dataclass
class ChangelogEntry:
    accession_no: str
    title: str
    form_type: str
    filed_date: str
    url: str | None
    summary: str          # LLM one-line "what's new"
    detected_at: str      # ISO timestamp (UTC)
```

---

## 4. ChangelogStore (`store/changelog_store.py`)

Wraps a Qdrant collection `regulation_changelog` (dense vector, 1024-dim cosine; no sparse needed). Reuses the existing `QdrantClient`.

- `ensure_collection()` — create if absent (dense vector only).
- `point_id(accession_no) -> str` — `uuid5` of the accession (deterministic, dedup).
- `is_seen(accession_no) -> bool` — `client.retrieve(ids=[point_id])` non-empty.
- `record(entry: ChangelogEntry, vector: list[float])` — upsert the point (payload = entry fields).
- `list_recent(limit=20) -> list[ChangelogEntry]` — scroll, sorted by `detected_at` desc.

---

## 5. MonitorAgent (`monitoring/agent.py`)

Constructor deps (all injected for testability): `sec_client`, `corpus_store` + `dense` + `sparse` (to ingest into `corpus` via `ingest_documents`), `changelog_store`, `dense` embedder (for the changelog vector), `provider` + `model` (for the summary).

`poll(query: str, *, forms: list[str], limit: int) -> list[ChangelogEntry]`:
1. `filings = sec_client.full_text_search(query, forms=forms, limit=limit)` (metadata only — cheap).
2. `new = [f for f in filings if f.doc_url and not changelog_store.is_seen(f.accession_no)]`.
3. For each `f` in `new` (each wrapped in try/except — skip on failure):
   - `body = sec_client.fetch_document(f.doc_url)`.
   - Ingest into `corpus`: build a `DocInput` (source=sec, jurisdiction=US-SEC, doc_type=filing, body text, metadata) and call `ingest_documents([doc], store=corpus_store, dense=dense, sparse=sparse)`.
   - `summary = provider.chat(...)` — one-line "what's new" from the filing title/body excerpt (chat tier; plain `chat`, not structured).
   - `vector = dense.embed_one(summary or title)`.
   - `entry = ChangelogEntry(..., detected_at=now_utc_iso())`; `changelog_store.record(entry, vector)`.
   - Append to results.
4. Return the recorded entries.

`now_utc_iso()` helper uses `datetime.now(timezone.utc).isoformat()`.

---

## 6. Scheduler (`monitoring/scheduler.py`)

- `run_scheduler(monitor, *, query, forms, limit, interval_seconds)` — APScheduler `BlockingScheduler`, one interval job calling `monitor.poll(...)`, wrapped so an exception in a poll is logged and does not kill the scheduler. Runs one poll immediately on start, then every `interval_seconds`.
- `build_default_monitor(settings)` — wires `SECClient`, `QdrantStore` (corpus), `OllamaEmbedder`, `BM25Encoder`, `ChangelogStore`, `OllamaProvider` from settings (mirrors `build_default_graph`).

---

## 7. CLI

- `regintel monitor --once [--query Q --forms 8-K,10-K --limit N]` — one poll; print each detected change (title, form, date, summary) or "no new filings."
- `regintel monitor [--interval 3600 ...]` — start the long-running scheduler.
- `regintel changelog [--limit 20]` — list recent recorded changes from the changelog.

---

## 8. Error handling

`poll()` per-filing try/except (skip + log on fetch/ingest/summary failure). Scheduler job wrapped: poll exceptions logged, scheduler continues. `ChangelogStore.ensure_collection()` is idempotent. SEC throttling/caching already handled in `SECClient`.

---

## 9. Testing (TDD)

- **ChangelogStore** (`tests/test_changelog_store.py`, in-memory `QdrantClient(":memory:")`): `ensure_collection` + `record` then `is_seen` true; unseen accession → false; `list_recent` returns recorded entries.
- **MonitorAgent** (`tests/test_monitor.py`, fakes): a fake SEC client returns 2 filings with `doc_url`; a fake corpus store, fake dense (returns fixed vector + has `embed_one`), fake sparse, fake changelog store (in-memory dict), fake provider (returns a summary). Assertions:
  - First `poll()` returns 2 entries; both ingested into corpus (corpus store received records); both recorded in changelog.
  - Second `poll()` (same filings) returns 0 (all seen).
  - A filing whose `fetch_document` raises is skipped (others still processed).
  - A filing without `doc_url` is skipped before any fetch.
- **Scheduler** (`tests/test_scheduler.py`): `run_scheduler` registers an interval job and the wrapped job swallows a poll exception (call the wrapped job function directly with a monitor whose `poll` raises; assert no exception propagates). Do not actually block.
- **Live** (`tests/test_monitor_live.py`, gated): real `monitor.poll(...)` against SEC with embedded Qdrant; assert it returns a list and (cold start) records ≥1 changelog entry; a second poll returns 0 new.

---

## 10. File structure

```
src/regintel/
  types.py                      # MODIFY: + ChangelogEntry
  store/changelog_store.py      # CREATE: ChangelogStore
  monitoring/__init__.py        # CREATE: empty
  monitoring/agent.py           # CREATE: MonitorAgent, now_utc_iso
  monitoring/scheduler.py       # CREATE: run_scheduler, build_default_monitor
  cli.py                        # MODIFY: + monitor, changelog commands
pyproject.toml                  # MODIFY: + apscheduler
tests/
  test_changelog_store.py  test_monitor.py  test_scheduler.py  test_monitor_live.py
```

---

## 11. Definition of done

1. `uv run pytest` green (new + existing); ruff clean.
2. `regintel monitor --once` polls SEC, ingests new filings into `corpus`, prints detected changes; a second run prints "no new filings."
3. `regintel changelog` lists recorded changes.
4. Newly monitored filings are retrievable by the Phase 2/3 `ask` pipeline (they're in `corpus`).
5. Scheduler runs polls on an interval and survives a poll error.

## 12. Out of scope

Auto impact-assessment on changes (deferred — keeps Monitor decoupled), content-level version diffing of the same filing, multi-source monitoring (SEC only), FastAPI/UI (Phase 5).
