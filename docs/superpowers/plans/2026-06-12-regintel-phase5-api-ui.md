# RegIntel Phase 5 Implementation Plan (FastAPI + Minimal Web UI)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Expose the existing `ask` pipeline + changelog over a FastAPI backend and a single static web page, so a reviewer can ask a compliance question and see a cited, self-evaluated report in the browser.

**Architecture:** `create_app(*, graph=None, changelog_store=None)` factory wraps `run_query` and `ChangelogStore` (DI for tests; lazy defaults otherwise). `POST /ask` (sync → threadpool) returns a serialized `Report`; `GET /changelog` returns recent entries; `GET /` serves a self-contained `index.html` (vanilla fetch, no build step). `regintel serve` runs uvicorn.

**Tech Stack:** FastAPI, uvicorn, existing `regintel` modules, pytest + Starlette `TestClient` (httpx already a dep).

**Conventions:** run via `uv run` (plain `uv run pytest`, no `--extra`); git author Jainum Sanghavi <sanghavi.h.j20@gmail.com>, NO Co-Authored-By trailer; commit per task; touch `pyproject.toml`/`uv.lock` ONLY in Task 1.

Reused facts (do not change):
- `regintel.orchestration.graph.run_query(query, *, graph) -> Report` calls `graph.invoke(new_state(query))["report"]` — so an injected fake `graph` only needs an `.invoke(state) -> dict` returning `{"report": <Report>}`.
- `regintel.orchestration.graph.build_default_graph(settings=None) -> compiled graph`.
- `regintel.store.changelog_store.ChangelogStore(client).list_recent(limit) -> list[ChangelogEntry]`.
- `regintel.types`: `Report(query_type: QueryType, answer, citations: list[Citation], findings: list[Finding], impacts: list[Impact], warnings, eval: EvalScores|None)`; `Citation(doc_id, chunk_index, title, source, url, quote)`; `Finding(topic, requirement, internal_status, gap, explanation, citations)`; `Impact(topic, affected_policies, severity, rationale)`; `EvalScores(faithfulness, citation_coverage, conflicts, flagged, notes)`; `ChangelogEntry(accession_no, title, form_type, filed_date, url, summary, detected_at)`; `QueryType` is a str-enum.
- `regintel.config.get_settings()`.

---

## File structure

```
src/regintel/api/__init__.py        # CREATE: empty
src/regintel/api/app.py             # CREATE: create_app, serializers, endpoints, module-level app
src/regintel/api/static/index.html  # CREATE: single-page UI
src/regintel/cli.py                 # MODIFY: + serve command
pyproject.toml                      # MODIFY: + fastapi, uvicorn
tests/test_api.py                   # CREATE
tests/test_api_live.py              # CREATE (gated)
```

---

## Task 1: FastAPI app — /ask, /changelog, /healthz + serializers

**Files:** Modify `pyproject.toml`, `uv.lock`; Create `src/regintel/api/__init__.py`, `src/regintel/api/app.py`, `tests/test_api.py`

- [ ] **Step 1: Add dependencies**

Run: `uv add fastapi uvicorn`
Verify: `uv run python -c "import fastapi, uvicorn; print('ok')"` → `ok`

- [ ] **Step 2: Write the failing test**

`tests/test_api.py`:
```python
from fastapi.testclient import TestClient

from regintel.api.app import create_app
from regintel.types import (
    Citation, EvalScores, Finding, Impact, QueryType, Report, ChangelogEntry,
)


def _report():
    return Report(
        query_type=QueryType.GAP_CHECK,
        answer="No. The policy lacks the SEC blackout window [0].",
        citations=[Citation("docu", 0, "DocuSign ITP", "sec", "http://x", "two trading days")],
        findings=[Finding(topic="blackout windows", requirement="define blackout",
                          internal_status="absent", gap=True, explanation="missing",
                          citations=[])],
        impacts=[Impact(topic="blackout windows", affected_policies=["ACME Policy"],
                        severity="high", rationale="material")],
        warnings=[],
        eval=EvalScores(faithfulness=0.86, citation_coverage=0.9, conflicts=[],
                        flagged=False, notes="ok"),
    )


class _FakeGraph:
    def __init__(self, report=None, exc=None):
        self._report = report
        self._exc = exc

    def invoke(self, state):
        if self._exc is not None:
            raise self._exc
        return {"report": self._report}


class _FakeChangelog:
    def __init__(self, entries):
        self._entries = entries

    def list_recent(self, limit=20):
        return self._entries[:limit]


def _client(graph=None, changelog=None):
    return TestClient(create_app(graph=graph or _FakeGraph(_report()),
                                 changelog_store=changelog or _FakeChangelog([])))


def test_healthz():
    assert _client().get("/healthz").json() == {"status": "ok"}


def test_ask_returns_serialized_report():
    r = _client().post("/ask", json={"query": "does our policy comply?"})
    assert r.status_code == 200
    body = r.json()
    assert body["query_type"] == "gap_check"
    assert "blackout window" in body["answer"]
    assert body["eval"]["faithfulness"] == 0.86
    assert body["eval"]["flagged"] is False
    assert len(body["citations"]) == 1
    assert body["citations"][0]["url"] == "http://x"
    assert body["findings"][0]["gap"] is True
    assert body["impacts"][0]["severity"] == "high"


def test_ask_empty_query_returns_400():
    r = _client().post("/ask", json={"query": "   "})
    assert r.status_code == 400


def test_ask_graph_error_returns_500():
    client = TestClient(create_app(graph=_FakeGraph(exc=RuntimeError("boom")),
                                   changelog_store=_FakeChangelog([])))
    r = client.post("/ask", json={"query": "x"})
    assert r.status_code == 500


def test_changelog_returns_entries():
    entries = [
        ChangelogEntry("a", "Filing A", "8-K", "2026-05-01", "http://a", "summary a",
                       "2026-06-12T00:00:00+00:00"),
        ChangelogEntry("b", "Filing B", "10-K", "2026-05-02", None, "summary b",
                       "2026-06-12T01:00:00+00:00"),
    ]
    r = _client(changelog=_FakeChangelog(entries)).get("/changelog?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["title"] == "Filing A"
    assert body[1]["url"] is None
```

- [ ] **Step 3: Run to verify fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL (ModuleNotFoundError: regintel.api.app).

- [ ] **Step 4: Create `src/regintel/api/__init__.py`** (empty file)

- [ ] **Step 5: Implement `src/regintel/api/app.py`**

```python
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from regintel.types import ChangelogEntry, Citation, Report

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX = _STATIC_DIR / "index.html"


class AskRequest(BaseModel):
    query: str


def _serialize_citation(c: Citation) -> dict:
    return {
        "doc_id": c.doc_id, "chunk_index": c.chunk_index, "title": c.title,
        "source": c.source, "url": c.url, "quote": c.quote,
    }


def _serialize_report(r: Report) -> dict:
    return {
        "query_type": r.query_type.value,
        "answer": r.answer,
        "citations": [_serialize_citation(c) for c in r.citations],
        "findings": [
            {
                "topic": f.topic, "requirement": f.requirement,
                "internal_status": f.internal_status, "gap": f.gap,
                "explanation": f.explanation,
                "citations": [_serialize_citation(c) for c in f.citations],
            }
            for f in r.findings
        ],
        "impacts": [
            {
                "topic": i.topic, "affected_policies": i.affected_policies,
                "severity": i.severity, "rationale": i.rationale,
            }
            for i in r.impacts
        ],
        "eval": (
            {
                "faithfulness": r.eval.faithfulness,
                "citation_coverage": r.eval.citation_coverage,
                "conflicts": r.eval.conflicts,
                "flagged": r.eval.flagged,
                "notes": r.eval.notes,
            }
            if r.eval is not None else None
        ),
        "warnings": r.warnings,
    }


def _serialize_changelog(e: ChangelogEntry) -> dict:
    return {
        "accession_no": e.accession_no, "title": e.title, "form_type": e.form_type,
        "filed_date": e.filed_date, "url": e.url, "summary": e.summary,
        "detected_at": e.detected_at,
    }


def create_app(*, graph=None, changelog_store=None) -> FastAPI:
    app = FastAPI(title="Regulatory Intelligence")
    state: dict = {"graph": graph, "changelog": changelog_store}

    def _get_graph():
        if state["graph"] is None:
            from regintel.orchestration.graph import build_default_graph
            state["graph"] = build_default_graph()
        return state["graph"]

    def _get_changelog():
        if state["changelog"] is None:
            from qdrant_client import QdrantClient
            from regintel.config import get_settings
            from regintel.store.changelog_store import ChangelogStore
            s = get_settings()
            client = (QdrantClient(path="./qdrant_storage") if s.qdrant_embedded
                      else QdrantClient(url=s.qdrant_url))
            store = ChangelogStore(client=client)
            store.ensure_collection()
            state["changelog"] = store
        return state["changelog"]

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        query = req.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        from regintel.orchestration.graph import run_query
        try:
            report = run_query(query, graph=_get_graph())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _serialize_report(report)

    @app.get("/changelog")
    def changelog(limit: int = 20) -> list[dict]:
        return [_serialize_changelog(e) for e in _get_changelog().list_recent(limit)]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_INDEX)

    return app


app = create_app()
```

> Note: the `GET /` route references `index.html`, created in Task 2. Task 1's tests do not hit `/`, so they pass without it.

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/regintel/api/__init__.py src/regintel/api/app.py tests/test_api.py
git commit -m "feat: add FastAPI app (/ask, /changelog, /healthz) with report serialization"
```

---

## Task 2: Static web UI (`api/static/index.html`)

**Files:** Create `src/regintel/api/static/index.html`; extend `tests/test_api.py`

- [ ] **Step 1: Add the failing test for `GET /`**

Append to `tests/test_api.py`:
```python
def test_index_serves_html():
    r = _client().get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Regulatory Intelligence" in r.text
    assert "/ask" in r.text  # the page wires the ask endpoint
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_api.py::test_index_serves_html -v`
Expected: FAIL (404 / file not found — index.html absent).

- [ ] **Step 3: Create `src/regintel/api/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Regulatory Intelligence</title>
<style>
  :root { --bg:#0f1419; --panel:#1a212b; --line:#2a3542; --fg:#e6edf3; --muted:#8b97a6;
          --accent:#4c8dff; --gap:#ff5d5d; --ok:#3fb950; --warn:#d29922; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:20px 28px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:20px; }
  header p { margin:4px 0 0; color:var(--muted); font-size:13px; }
  .wrap { display:grid; grid-template-columns:1fr 320px; gap:24px; padding:24px 28px; max-width:1200px; }
  @media (max-width:880px){ .wrap{ grid-template-columns:1fr; } }
  .ask { display:flex; gap:8px; margin-bottom:20px; }
  .ask input { flex:1; padding:12px 14px; border-radius:8px; border:1px solid var(--line);
               background:var(--panel); color:var(--fg); font-size:15px; }
  .ask button { padding:12px 20px; border:0; border-radius:8px; background:var(--accent);
                color:#fff; font-weight:600; cursor:pointer; }
  .ask button:disabled { opacity:.5; cursor:default; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px; margin-bottom:16px; }
  .badge { display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px; font-weight:700;
           letter-spacing:.04em; text-transform:uppercase; }
  .badge.type { background:#23303f; color:var(--accent); }
  .badge.gap { background:#3a1f24; color:var(--gap); }
  .badge.ok { background:#16301f; color:var(--ok); }
  .badge.sev-high,.badge.sev-critical { background:#3a1f24; color:var(--gap); }
  .badge.sev-medium { background:#332a14; color:var(--warn); }
  .badge.sev-low { background:#16301f; color:var(--ok); }
  .answer { white-space:pre-wrap; margin:12px 0; }
  .evalbar { display:flex; gap:18px; align-items:center; font-size:13px; color:var(--muted);
             border-top:1px solid var(--line); padding-top:12px; margin-top:6px; }
  .evalbar b { color:var(--fg); }
  h3 { font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:18px 0 8px; }
  ul { margin:0; padding-left:0; list-style:none; }
  li { margin:8px 0; }
  .cite a { color:var(--accent); text-decoration:none; }
  .src { font-size:11px; color:var(--muted); }
  aside h2 { font-size:14px; margin:0 0 12px; }
  .change { border-bottom:1px solid var(--line); padding:10px 0; font-size:13px; }
  .change .t { font-weight:600; }
  .change .m { color:var(--muted); font-size:12px; }
  .spinner { display:none; color:var(--muted); }
  .error { color:var(--gap); }
  .muted { color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Regulatory Intelligence</h1>
  <p>Ask a compliance question — get a cited, self-evaluated report over live SEC filings + internal policies.</p>
</header>
<div class="wrap">
  <main>
    <div class="ask">
      <input id="q" placeholder="e.g. Does our insider trading policy comply with SEC blackout rules?"
             autocomplete="off" />
      <button id="go">Ask</button>
    </div>
    <div id="spinner" class="spinner">Analyzing… (running the multi-agent pipeline, ~30–90s)</div>
    <div id="error" class="error"></div>
    <div id="result"></div>
  </main>
  <aside>
    <h2>Recent regulatory changes</h2>
    <div id="changes" class="muted">Loading…</div>
  </aside>
</div>
<script>
const $ = (id) => document.getElementById(id);

function badge(cls, text) { return `<span class="badge ${cls}">${text}</span>`; }
function esc(s) { return (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function ask() {
  const q = $("q").value.trim();
  if (!q) return;
  $("go").disabled = true; $("spinner").style.display = "block";
  $("error").textContent = ""; $("result").innerHTML = "";
  try {
    const res = await fetch("/ask", {method:"POST", headers:{"Content-Type":"application/json"},
                                     body: JSON.stringify({query:q})});
    if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e.detail || res.status); }
    render(await res.json());
  } catch (err) {
    $("error").textContent = "Error: " + err.message;
  } finally {
    $("go").disabled = false; $("spinner").style.display = "none";
  }
}

function render(r) {
  const ev = r.eval;
  let evalbar = "";
  if (ev) {
    const flag = ev.flagged ? badge("gap","⚠ flagged") : badge("ok","✓ verified");
    evalbar = `<div class="evalbar">
      <span>Faithfulness <b>${(ev.faithfulness*100).toFixed(0)}%</b></span>
      <span>Citation coverage <b>${(ev.citation_coverage*100).toFixed(0)}%</b></span>
      <span>Conflicts <b>${ev.conflicts.length}</b></span>${flag}</div>`;
  }
  const cites = r.citations.map((c,i)=>`<li class="cite">[${i}] <span class="src">(${esc(c.source)})</span>
      ${c.url ? `<a href="${esc(c.url)}" target="_blank">${esc(c.title)} ↗</a>` : esc(c.title)}</li>`).join("");
  const findings = r.findings.map(f=>`<li>${badge(f.gap?"gap":"ok", f.gap?"gap":"ok")}
      <b>${esc(f.topic)}</b> — ${esc(f.explanation)}</li>`).join("");
  const impacts = r.impacts.map(i=>`<li>${badge("sev-"+esc(i.severity), esc(i.severity))}
      <b>${esc(i.topic)}</b> — ${esc((i.affected_policies||[]).join(", "))}</li>`).join("");
  $("result").innerHTML = `<div class="card">
      ${badge("type", esc(r.query_type))}
      <div class="answer">${esc(r.answer)}</div>${evalbar}</div>
    ${cites ? `<h3>Citations</h3><ul>${cites}</ul>`:""}
    ${findings ? `<h3>Findings</h3><ul>${findings}</ul>`:""}
    ${impacts ? `<h3>Impacts</h3><ul>${impacts}</ul>`:""}`;
}

async function loadChanges() {
  try {
    const res = await fetch("/changelog?limit=10");
    const items = await res.json();
    if (!items.length) { $("changes").textContent = "No changes recorded yet."; return; }
    $("changes").innerHTML = items.map(e=>`<div class="change">
        <div class="t">${badge("type", esc(e.form_type))} ${esc(e.title)}</div>
        <div class="m">${esc(e.filed_date)}</div>
        <div>${esc(e.summary)}</div>
        ${e.url ? `<a href="${esc(e.url)}" target="_blank" class="src">view filing ↗</a>`:""}</div>`).join("");
  } catch { $("changes").textContent = "Changelog unavailable."; }
}

$("go").addEventListener("click", ask);
$("q").addEventListener("keydown", e => { if (e.key === "Enter") ask(); });
loadChanges();
</script>
</body>
</html>
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/regintel/api/static/index.html tests/test_api.py
git commit -m "feat: add single-page web UI served at /"
```

---

## Task 3: `serve` CLI command + live test

**Files:** Modify `src/regintel/cli.py`; Create `tests/test_api_live.py`

- [ ] **Step 1: Add the `serve` command to `src/regintel/cli.py`**

Add this function after `cmd_changelog`:
```python
def cmd_serve(args) -> None:
    import uvicorn
    from regintel.api.app import create_app
    uvicorn.run(create_app(), host=args.host, port=args.port)
```
Register it in `main()` (after the `changelog` subparser block, before `args = parser.parse_args()`):
```python
    p_srv = sub.add_parser("serve")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.set_defaults(func=cmd_serve)
```

- [ ] **Step 2: Verify the CLI parses**

Run: `uv run python -m regintel.cli serve --help`
Expected: usage text showing `--host` and `--port` (does not start the server).

- [ ] **Step 3: Add the gated live test `tests/test_api_live.py`**

```python
import pytest


@pytest.mark.live
def test_ask_endpoint_live():
    """Requires Ollama + (embedded) Qdrant with ingested data. Uses default wiring."""
    from fastapi.testclient import TestClient
    from regintel.api.app import create_app

    client = TestClient(create_app())
    r = client.post("/ask", json={"query": "What are insider trading blackout window rules?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["query_type"] in {"lookup", "gap_check", "impact"}
    assert body["eval"] is None or 0.0 <= body["eval"]["faithfulness"] <= 1.0
```

- [ ] **Step 4: Full suite + lint**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all non-live tests PASS; ruff clean. Fix any lint.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/cli.py tests/test_api_live.py
git commit -m "feat: add `serve` CLI command and gated live API test"
```

- [ ] **Step 6 (controller): live smoke**

Start the server and exercise it (requires Ollama + ingested corpus):
```bash
QDRANT_EMBEDDED=true uv run python -m regintel.cli serve &
sleep 3
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"Does our insider trading policy comply with SEC blackout rules?"}' | head -c 400
```
Expected: healthz ok; /ask returns a JSON report. Stop the background server afterward.

---

## Self-Review (completed by planner)

**Spec coverage:**
- create_app factory + DI → Task 1 ✅
- POST /ask sync + 400 empty + 500 on error + serialized report → Task 1 ✅
- GET /changelog → Task 1 ✅
- GET /healthz → Task 1 ✅
- _serialize_report / _serialize_changelog (enum .value, nested dataclasses) → Task 1 ✅
- GET / serves static index.html → Task 2 ✅
- Single-page UI (query box, query-type badge, answer, eval strip, citations, findings, impacts, changes panel, spinner, error) → Task 2 ✅
- serve CLI + module-level `app` → Task 1 (`app = create_app()`) + Task 3 ✅
- fastapi/uvicorn deps → Task 1 ✅
- Tests (ask success/empty/error, changelog, /, healthz; gated live) → Tasks 1-3 ✅

**Placeholder scan:** none — all code complete.

**Type consistency:** `_FakeGraph.invoke(state) -> {"report": Report}` matches `run_query`'s `graph.invoke(...)["report"]`. Serializer field names match the `types.py` dataclasses exactly (Citation/Finding/Impact/EvalScores/ChangelogEntry). `create_app(*, graph=None, changelog_store=None)` keyword set identical in Task 1 tests, Task 2 test helper, Task 3 default call, and `app = create_app()`. `ChangelogEntry(...)` positional order in test (`"a","Filing A","8-K","2026-05-01","http://a","summary a","<ts>"`) matches the dataclass field order (accession_no, title, form_type, filed_date, url, summary, detected_at). The UI's `/ask` substring satisfies `test_index_serves_html`.

**Note:** `app = create_app()` at import builds NO network resources (graph/changelog are lazy via `_get_*`), so importing `regintel.api.app` in tests is cheap and offline.
```
