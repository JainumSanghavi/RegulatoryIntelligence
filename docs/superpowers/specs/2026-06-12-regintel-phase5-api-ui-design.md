# Regulatory Intelligence System — Phase 5 Design (FastAPI + Minimal Web UI)

**Date:** 2026-06-12
**Status:** Approved (brainstorming) — continuous execution authorized
**Scope:** Phase 5 (final) — a FastAPI backend wrapping the existing `ask` pipeline + changelog, and a single static web page so a reviewer can ask a compliance question and see a cited, self-evaluated report live. Builds on Phases 2–4.

---

## 1. Context

Phases 0–4 deliver the full engine (retrieval → orchestration → reasoning → evaluation → monitoring), driven by a CLI. Phase 5 puts a thin web surface on it: a "click and see it work" demo that turns the CLI into something a non-technical reviewer can use in a browser. This is the portfolio finish line.

---

## 2. Decisions (locked in brainstorming)

- **Static page served by FastAPI** — one hand-written `index.html` (vanilla `fetch`, no build step, no `node_modules`, zero added disk). Reliable for a live demo.
- **Read-only changelog panel** — the UI lists recent detected changes from `GET /changelog` (populated by the `monitor` CLI/scheduler). No slow live poll triggered from the browser.
- Thin layer: the API wraps existing `run_query` / `ChangelogStore`; no new business logic.

---

## 3. API (`src/regintel/api/app.py`)

`create_app(*, graph=None, changelog_store=None) -> FastAPI` — app factory. Dependency-injected `graph` and `changelog_store` (so tests inject fakes); when omitted, lazily build the defaults (`build_default_graph(settings)` and a `ChangelogStore` on the settings' Qdrant client).

Endpoints:
- `GET /` → serve `api/static/index.html` (FileResponse).
- `POST /ask`, body `{"query": str}` → **sync def** (FastAPI runs it in a threadpool so the ~30–90s pipeline doesn't block the event loop). Empty/missing query → `400`. Runs `run_query(query, graph=graph)`; on success returns `_serialize_report(report)`; on exception → `500` with `{"detail": str(exc)}`.
- `GET /changelog?limit=20` → `[_serialize_changelog(e) for e in changelog_store.list_recent(limit)]`.
- `GET /healthz` → `{"status": "ok"}`.
- Static files (the single HTML) served via a `StaticFiles` mount or a direct `FileResponse` at `/`.

`_serialize_report(report: Report) -> dict`: explicit dict with `query_type` (`.value`), `answer`, `citations` (list of `{doc_id, chunk_index, title, source, url, quote}`), `findings` (`{topic, requirement, internal_status, gap, explanation, citations}`), `impacts` (`{topic, affected_policies, severity, rationale}`), `eval` (`{faithfulness, citation_coverage, conflicts, flagged, notes}` or `null`), `warnings`. Avoids relying on enum auto-encoding.

`_serialize_changelog(e: ChangelogEntry) -> dict`: `{accession_no, title, form_type, filed_date, url, summary, detected_at}`.

---

## 4. Web UI (`src/regintel/api/static/index.html`)

Single self-contained file (inline CSS + JS, vanilla `fetch`). Layout: a header, a two-column body — main column (query box + results) and a right panel (recent changes). Polished, minimal, no framework.

- **Query box** + Ask button → `POST /ask`. Shows a spinner while pending; inline error message on failure.
- **Result** renders: query-type badge; the answer text (with `[n]` markers shown as-is); an **Evaluation** strip (`faithfulness`, `citation_coverage` as percentages, conflict count, a green "verified" or amber "⚠ flagged" badge); **Citations** (numbered, source badge, title linked to `url` when present); **Findings** (topic + GAP/ok badge + explanation); **Impacts** (topic + severity badge + affected policies).
- **Recent changes panel**: on load, `GET /changelog?limit=10` → list of `{form_type, title, filed_date, summary, url}`.
- Empty state and error state handled in JS.

---

## 5. CLI / entry point

Add `regintel serve [--host 127.0.0.1] [--port 8000]` → runs `uvicorn.run(create_app(), host, port)`. Document `uvicorn regintel.api.app:app` as an alternative (an module-level `app = create_app()` is exposed for that).

Adds `fastapi` and `uvicorn` to dependencies. (`httpx` — needed by `TestClient` — is already a dep.)

---

## 6. Error handling

- Empty query → `400`.
- `run_query` raising → `500` with the error string; the UI surfaces it inline (the pipeline itself rarely raises — it degrades into the report's `warnings` — but the endpoint is defensive).
- `/changelog` when the collection is empty/missing → `ensure_collection()` then return `[]`.
- The static file is shipped in the package; `/` returns it directly.

---

## 7. Testing (TDD)

`tests/test_api.py` with FastAPI `TestClient` and injected fakes (no live models):
- A fake graph whose `invoke` (or a stub passed to `create_app`) yields a canned `Report` (gap-check with one citation, finding, impact, and `EvalScores`). Inject via `create_app(graph=fake_graph, changelog_store=fake_changelog)`.
- `POST /ask {"query": "..."}` → 200; body has `query_type == "gap_check"`, non-empty `answer`, `eval.faithfulness` a float, ≥1 citation.
- `POST /ask {"query": ""}` → 400.
- `POST /ask` when the graph raises → 500.
- `GET /changelog` → 200, list of entries with expected fields (fake changelog returns 2).
- `GET /` → 200, `text/html`, body contains the app title.
- `GET /healthz` → 200 `{"status": "ok"}`.
- (Gated `live`) `tests/test_api_live.py`: `create_app()` default wiring + real `POST /ask` returns a report with an `eval`. Deselected by default.

`run_query(query, *, graph)` already exists; the fake graph must satisfy `graph.invoke(state)` returning a state dict whose `["report"]` is the canned `Report` (matching how `run_query` reads it). The API tests inject a `graph` object with an `invoke` method.

---

## 8. File structure

```
src/regintel/
  api/__init__.py             # CREATE: empty
  api/app.py                  # CREATE: create_app, endpoints, serializers, module-level app
  api/static/index.html       # CREATE: single-page UI
  cli.py                      # MODIFY: + serve command
pyproject.toml                # MODIFY: + fastapi, uvicorn
tests/
  test_api.py                 # CREATE
  test_api_live.py            # CREATE (gated)
```

---

## 9. Definition of done

1. `uv run pytest` green (new + existing); ruff clean.
2. `regintel serve` starts; `GET /` shows the page; asking a question returns a rendered cited+evaluated report; the changes panel lists changelog entries.
3. API responses are JSON-serializable (no enum/dataclass errors).
4. Tests cover /ask (success, empty→400, error→500), /changelog, /, /healthz with injected fakes.

## 10. Out of scope

Authentication, multi-user concurrency/streaming, a live "poll now" button (changelog is read-only), a build-step SPA, deployment/hosting config. This completes the 6-phase build.
