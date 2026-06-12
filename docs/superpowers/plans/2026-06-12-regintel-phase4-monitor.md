# RegIntel Phase 4 Implementation Plan (Monitor / Scheduled Change Detection)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a scheduled, query-decoupled MonitorAgent that polls SEC EDGAR, detects unseen filings, ingests them into the queryable `corpus`, and records a changelog entry per change.

**Architecture:** A `regulation_changelog` Qdrant collection records seen filings (keyed by accession). `MonitorAgent.poll()` searches SEC, filters out already-seen filings, ingests new ones via the existing `ingest_documents`, summarizes each with the LLM, and records it. An APScheduler `BlockingScheduler` runs polls on an interval; CLI exposes `monitor` and `changelog`. Reuses SECClient, QdrantStore, OllamaEmbedder, BM25Encoder, OllamaProvider.

**Tech Stack:** APScheduler, qdrant-client, existing `regintel` modules, pytest.

**Conventions:** run via `uv run` (plain `uv run pytest`, no `--extra`); git author Jainum Sanghavi <sanghavi.h.j20@gmail.com>, NO Co-Authored-By trailer; commit per task; touch `pyproject.toml`/`uv.lock` ONLY in Task 1.

---

## File structure

```
src/regintel/
  types.py                      # MODIFY: + ChangelogEntry
  store/changelog_store.py      # CREATE: ChangelogStore
  monitoring/__init__.py        # CREATE: empty
  monitoring/agent.py           # CREATE: MonitorAgent + now_utc_iso
  monitoring/scheduler.py       # CREATE: make_poll_job, run_scheduler, build_default_monitor
  cli.py                        # MODIFY: + monitor, changelog commands
tests/
  test_changelog_store.py  test_monitor.py  test_scheduler.py  test_monitor_live.py
```

Reused signatures (do not change them):
- `regintel.ingest.pipeline.DocInput(doc_id, title, text, source, jurisdiction, doc_type, url=None, regulation_id=None, form_type=None, accession_no=None, effective_date=None, filed_date=None)`
- `regintel.ingest.pipeline.ingest_documents(docs, *, store, dense, sparse, chunk_tokens=800, overlap_tokens=150) -> int` — calls `store.ensure_collection()`, `dense.embed(list[str])`, `sparse.encode(list[str])`, `store.upsert(records)`.
- `regintel.ingest.sec_edgar.SECClient.full_text_search(query, *, forms=None, limit=10) -> list[SECFiling]`; `.fetch_document(url) -> str`. `SECFiling(accession_no, title, form_type, filed_date, cik=None, doc_url=None)`.
- `regintel.embeddings.ollama_embedder.OllamaEmbedder.embed_one(text) -> list[float]` and `.embed(list[str])`.
- `regintel.llm.ollama_provider.OllamaProvider.chat(messages, *, model=None) -> str`; `regintel.llm.base.ChatMessage`.
- `regintel.store.schema.CHANGELOG_COLLECTION` (= "regulation_changelog"), `DENSE_DIM` (= 1024).

---

## Task 1: apscheduler dependency + `ChangelogEntry` type

**Files:** Modify `pyproject.toml`, `uv.lock`, `src/regintel/types.py`; Test `tests/test_changelog_store.py` (types portion)

- [ ] **Step 1: Add the dependency**

Run: `uv add apscheduler`
Verify: `uv run python -c "from apscheduler.schedulers.blocking import BlockingScheduler; print('ok')"` → `ok`

- [ ] **Step 2: Write the failing test**

`tests/test_changelog_store.py`:
```python
from regintel.types import ChangelogEntry


def test_changelog_entry_fields():
    e = ChangelogEntry(accession_no="0001-24-1", title="ACME 8-K", form_type="8-K",
                       filed_date="2026-05-01", url="http://x", summary="new blackout rule",
                       detected_at="2026-06-12T00:00:00+00:00")
    assert e.accession_no == "0001-24-1"
    assert e.summary == "new blackout rule"
    assert e.url == "http://x"
```

- [ ] **Step 3: Run to verify fail**

Run: `uv run pytest tests/test_changelog_store.py -v`
Expected: FAIL (ImportError: cannot import name 'ChangelogEntry').

- [ ] **Step 4: Append `ChangelogEntry` to `src/regintel/types.py`**

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

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_changelog_store.py -v`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/regintel/types.py tests/test_changelog_store.py
git commit -m "feat: add apscheduler dep and ChangelogEntry type"
```

---

## Task 2: ChangelogStore (`store/changelog_store.py`)

**Files:** Create `src/regintel/store/changelog_store.py`; extend `tests/test_changelog_store.py`

- [ ] **Step 1: Append failing tests to `tests/test_changelog_store.py`**

```python
from qdrant_client import QdrantClient

from regintel.store.changelog_store import ChangelogStore


def _store():
    s = ChangelogStore(client=QdrantClient(":memory:"))
    s.ensure_collection()
    return s


def _entry(acc, detected_at):
    return ChangelogEntry(accession_no=acc, title=f"Filing {acc}", form_type="8-K",
                          filed_date="2026-05-01", url=f"http://x/{acc}",
                          summary=f"summary {acc}", detected_at=detected_at)


def test_record_then_is_seen():
    s = _store()
    assert s.is_seen("acc1") is False
    s.record(_entry("acc1", "2026-06-12T00:00:00+00:00"), vector=[0.1] * 1024)
    assert s.is_seen("acc1") is True
    assert s.is_seen("acc2") is False


def test_record_is_idempotent():
    s = _store()
    e = _entry("acc1", "2026-06-12T00:00:00+00:00")
    s.record(e, vector=[0.1] * 1024)
    s.record(e, vector=[0.1] * 1024)
    assert len(s.list_recent()) == 1


def test_list_recent_sorted_desc_by_detected_at():
    s = _store()
    s.record(_entry("a", "2026-06-10T00:00:00+00:00"), vector=[0.1] * 1024)
    s.record(_entry("b", "2026-06-12T00:00:00+00:00"), vector=[0.2] * 1024)
    s.record(_entry("c", "2026-06-11T00:00:00+00:00"), vector=[0.3] * 1024)
    recent = s.list_recent(limit=2)
    assert [e.accession_no for e in recent] == ["b", "c"]
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_changelog_store.py -v`
Expected: FAIL (ModuleNotFoundError: regintel.store.changelog_store).

- [ ] **Step 3: Implement `src/regintel/store/changelog_store.py`**

```python
import uuid
from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from regintel.store.schema import CHANGELOG_COLLECTION, DENSE_DIM
from regintel.types import ChangelogEntry

_NAMESPACE = uuid.UUID("22222222-2222-2222-2222-222222222222")


class ChangelogStore:
    """Records detected SEC filings in the regulation_changelog collection."""

    def __init__(self, client: QdrantClient, collection: str = CHANGELOG_COLLECTION) -> None:
        self._client = client
        self._collection = collection

    @staticmethod
    def point_id(accession_no: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, accession_no))

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE),
        )

    def is_seen(self, accession_no: str) -> bool:
        got = self._client.retrieve(self._collection, ids=[self.point_id(accession_no)])
        return len(got) > 0

    def record(self, entry: ChangelogEntry, vector: list[float]) -> None:
        self._client.upsert(
            self._collection,
            points=[qm.PointStruct(id=self.point_id(entry.accession_no),
                                   vector=vector, payload=asdict(entry))],
        )

    def list_recent(self, limit: int = 20) -> list[ChangelogEntry]:
        points, _ = self._client.scroll(self._collection, limit=10_000,
                                        with_payload=True, with_vectors=False)
        entries = [ChangelogEntry(**p.payload) for p in points]
        entries.sort(key=lambda e: e.detected_at, reverse=True)
        return entries[:limit]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_changelog_store.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/regintel/store/changelog_store.py tests/test_changelog_store.py
git commit -m "feat: add ChangelogStore (seen-tracking + recorded changes)"
```

---

## Task 3: MonitorAgent (`monitoring/agent.py`)

**Files:** Create `src/regintel/monitoring/__init__.py` (empty), `src/regintel/monitoring/agent.py`; Test `tests/test_monitor.py`

- [ ] **Step 1: Write the failing test**

`tests/test_monitor.py`:
```python
from regintel.monitoring.agent import MonitorAgent, now_utc_iso
from regintel.ingest.sec_edgar import SECFiling
from regintel.embeddings.sparse import SparseVec


class _FakeSEC:
    def __init__(self, filings, bodies, raise_for=None):
        self._filings = filings
        self._bodies = bodies
        self._raise_for = raise_for or set()
        self.fetched = []

    def full_text_search(self, query, *, forms=None, limit=10):
        return self._filings[:limit]

    def fetch_document(self, url):
        self.fetched.append(url)
        if url in self._raise_for:
            raise RuntimeError("fetch failed")
        return self._bodies[url]


class _FakeCorpus:
    def __init__(self):
        self.records = []
        self.ensured = False
    def ensure_collection(self):
        self.ensured = True
    def upsert(self, records):
        self.records.extend(records)


class _FakeDense:
    def embed(self, texts):
        return [[0.1] * 1024 for _ in texts]
    def embed_one(self, text):
        return [0.2] * 1024


class _FakeSparse:
    def encode(self, texts):
        return [SparseVec([1], [1.0]) for _ in texts]


class _FakeChangelog:
    def __init__(self):
        self.seen = {}
    def is_seen(self, accession_no):
        return accession_no in self.seen
    def record(self, entry, vector):
        self.seen[entry.accession_no] = entry


class _FakeProvider:
    def chat(self, messages, *, model=None, temperature=0.0, **kw):
        return "One-line summary."


def _filing(acc, url):
    return SECFiling(accession_no=acc, title=f"Filing {acc}", form_type="8-K",
                     filed_date="2026-05-01", cik="1", doc_url=url)


def _monitor(sec, changelog):
    return MonitorAgent(
        sec_client=sec, corpus_store=_FakeCorpus(), dense=_FakeDense(),
        sparse=_FakeSparse(), changelog_store=changelog,
        provider=_FakeProvider(), summary_model="m",
    )


def test_now_utc_iso_has_offset():
    assert "+00:00" in now_utc_iso() or now_utc_iso().endswith("Z")


def test_poll_detects_ingests_and_records_new_filings():
    sec = _FakeSEC([_filing("a", "http://x/a"), _filing("b", "http://x/b")],
                   {"http://x/a": "body a", "http://x/b": "body b"})
    changelog = _FakeChangelog()
    corpus = _FakeCorpus()
    monitor = MonitorAgent(sec_client=sec, corpus_store=corpus, dense=_FakeDense(),
                           sparse=_FakeSparse(), changelog_store=changelog,
                           provider=_FakeProvider(), summary_model="m")
    entries = monitor.poll("insider trading", forms=["8-K"], limit=10)
    assert {e.accession_no for e in entries} == {"a", "b"}
    assert corpus.ensured is True
    assert len(corpus.records) >= 2          # both filings ingested into corpus
    assert set(changelog.seen) == {"a", "b"}  # both recorded
    assert all(e.summary == "One-line summary." for e in entries)
    assert all(e.detected_at for e in entries)


def test_poll_skips_already_seen():
    sec = _FakeSEC([_filing("a", "http://x/a")], {"http://x/a": "body a"})
    changelog = _FakeChangelog()
    monitor = _monitor(sec, changelog)
    first = monitor.poll("q", forms=["8-K"], limit=10)
    assert len(first) == 1
    second = monitor.poll("q", forms=["8-K"], limit=10)
    assert second == []


def test_poll_skips_filing_without_url():
    sec = _FakeSEC([SECFiling("a", "No URL", "8-K", "2026-05-01", cik=None, doc_url=None)], {})
    monitor = _monitor(sec, _FakeChangelog())
    assert monitor.poll("q", forms=["8-K"], limit=10) == []
    assert sec.fetched == []  # never attempted a fetch


def test_poll_skips_filing_whose_fetch_fails():
    sec = _FakeSEC([_filing("a", "http://x/a"), _filing("b", "http://x/b")],
                   {"http://x/b": "body b"}, raise_for={"http://x/a"})
    changelog = _FakeChangelog()
    monitor = _monitor(sec, changelog)
    entries = monitor.poll("q", forms=["8-K"], limit=10)
    assert [e.accession_no for e in entries] == ["b"]  # "a" raised, skipped
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: FAIL (ModuleNotFoundError: regintel.monitoring.agent).

- [ ] **Step 3: Implement `src/regintel/monitoring/__init__.py`** (empty file)

- [ ] **Step 4: Implement `src/regintel/monitoring/agent.py`**

```python
import logging
from datetime import datetime, timezone

from regintel.ingest.pipeline import DocInput, ingest_documents
from regintel.llm.base import ChatMessage
from regintel.types import ChangelogEntry

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = (
    "Summarize in ONE concise sentence what this SEC filing is about and why it might "
    "matter for compliance. No preamble."
)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MonitorAgent:
    def __init__(self, *, sec_client, corpus_store, dense, sparse, changelog_store,
                 provider, summary_model: str) -> None:
        self._sec = sec_client
        self._corpus = corpus_store
        self._dense = dense
        self._sparse = sparse
        self._changelog = changelog_store
        self._provider = provider
        self._summary_model = summary_model

    def poll(self, query: str, *, forms: list[str], limit: int = 10) -> list[ChangelogEntry]:
        filings = self._sec.full_text_search(query, forms=forms, limit=limit)
        recorded: list[ChangelogEntry] = []
        for f in filings:
            if not f.doc_url or self._changelog.is_seen(f.accession_no):
                continue
            try:
                body = self._sec.fetch_document(f.doc_url)
                doc = DocInput(
                    doc_id=f.accession_no, title=f.title, text=f"{f.title}\n\n{body}",
                    source="sec", jurisdiction="US-SEC", doc_type="filing",
                    url=f.doc_url, form_type=f.form_type, accession_no=f.accession_no,
                    filed_date=f.filed_date,
                )
                ingest_documents([doc], store=self._corpus, dense=self._dense, sparse=self._sparse)
                summary = self._summarize(f.title, body)
                vector = self._dense.embed_one(summary or f.title)
                entry = ChangelogEntry(
                    accession_no=f.accession_no, title=f.title, form_type=f.form_type,
                    filed_date=f.filed_date, url=f.doc_url, summary=summary,
                    detected_at=now_utc_iso(),
                )
                self._changelog.record(entry, vector)
                recorded.append(entry)
            except Exception as exc:  # noqa: BLE001 - skip a bad filing, keep polling
                logger.warning("Monitor skipping %s: %s", f.accession_no, exc)
                continue
        return recorded

    def _summarize(self, title: str, body: str) -> str:
        try:
            text = self._provider.chat(
                [ChatMessage("system", _SUMMARY_SYSTEM),
                 ChatMessage("user", f"{title}\n\n{body[:2000]}")],
                model=self._summary_model,
            )
            return text.strip()
        except Exception as exc:  # noqa: BLE001 - summary is non-critical
            logger.warning("summary failed for %s: %s", title, exc)
            return title
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add src/regintel/monitoring/__init__.py src/regintel/monitoring/agent.py tests/test_monitor.py
git commit -m "feat: add MonitorAgent (detect new SEC filings -> ingest -> changelog)"
```

---

## Task 4: Scheduler + CLI + live test

**Files:** Create `src/regintel/monitoring/scheduler.py`, `tests/test_scheduler.py`, `tests/test_monitor_live.py`; Modify `src/regintel/cli.py`

- [ ] **Step 1: Write the failing scheduler test**

`tests/test_scheduler.py`:
```python
from regintel.monitoring.scheduler import make_poll_job


class _OkMonitor:
    def __init__(self):
        self.calls = 0
    def poll(self, query, *, forms, limit):
        self.calls += 1
        return ["entry"]


class _RaisingMonitor:
    def poll(self, query, *, forms, limit):
        raise RuntimeError("poll boom")


def test_poll_job_calls_monitor():
    m = _OkMonitor()
    job = make_poll_job(m, query="q", forms=["8-K"], limit=5)
    job()
    assert m.calls == 1


def test_poll_job_swallows_exceptions():
    job = make_poll_job(_RaisingMonitor(), query="q", forms=["8-K"], limit=5)
    job()  # must not raise
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: FAIL (ModuleNotFoundError: regintel.monitoring.scheduler).

- [ ] **Step 3: Implement `src/regintel/monitoring/scheduler.py`**

```python
import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from regintel.config import Settings, get_settings

logger = logging.getLogger(__name__)


def make_poll_job(monitor, *, query: str, forms: list[str], limit: int):
    def job() -> None:
        try:
            entries = monitor.poll(query, forms=forms, limit=limit)
            logger.info("Monitor poll: %d new filing(s)", len(entries))
        except Exception as exc:  # noqa: BLE001 - never kill the scheduler
            logger.error("Monitor poll failed: %s", exc)
    return job


def run_scheduler(monitor, *, query: str, forms: list[str], limit: int,
                  interval_seconds: int) -> None:
    job = make_poll_job(monitor, query=query, forms=forms, limit=limit)
    job()  # run once immediately on start
    scheduler = BlockingScheduler()
    scheduler.add_job(job, "interval", seconds=interval_seconds)
    logger.info("Monitor scheduler started (every %ds)", interval_seconds)
    scheduler.start()


def build_default_monitor(settings: Settings | None = None, *, client=None):
    settings = settings or get_settings()
    from qdrant_client import QdrantClient

    from regintel.embeddings.ollama_embedder import OllamaEmbedder
    from regintel.embeddings.sparse import BM25Encoder
    from regintel.ingest.sec_edgar import SECClient
    from regintel.llm.ollama_provider import OllamaProvider
    from regintel.monitoring.agent import MonitorAgent
    from regintel.store.changelog_store import ChangelogStore
    from regintel.store.qdrant_store import QdrantStore
    from pathlib import Path

    if client is None:
        client = (QdrantClient(path="./qdrant_storage") if settings.qdrant_embedded
                  else QdrantClient(url=settings.qdrant_url))
    corpus = QdrantStore(client=client)
    corpus.ensure_collection()
    changelog = ChangelogStore(client=client)
    changelog.ensure_collection()
    return MonitorAgent(
        sec_client=SECClient(user_agent=settings.sec_user_agent, cache_dir=Path("data/cache")),
        corpus_store=corpus,
        dense=OllamaEmbedder(host=settings.ollama_host, model=settings.ollama_embed_model),
        sparse=BM25Encoder(),
        changelog_store=changelog,
        provider=OllamaProvider(host=settings.ollama_host, default_model=settings.ollama_chat_model),
        summary_model=settings.ollama_chat_model,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Add `monitor` and `changelog` commands to `src/regintel/cli.py`**

Add imports near the others:
```python
from regintel.monitoring.scheduler import build_default_monitor, run_scheduler
from regintel.store.changelog_store import ChangelogStore
```
Add these functions after `cmd_ask`:
```python
def cmd_monitor(args) -> None:
    settings = get_settings()
    monitor = build_default_monitor(settings)
    forms = [f.strip() for f in args.forms.split(",") if f.strip()]
    if args.once:
        entries = monitor.poll(args.query, forms=forms, limit=args.limit)
        if not entries:
            print("No new filings.")
            return
        print(f"Detected {len(entries)} new filing(s):")
        for e in entries:
            print(f"  - [{e.form_type}] {e.title} ({e.filed_date})\n      {e.summary}")
    else:
        run_scheduler(monitor, query=args.query, forms=forms, limit=args.limit,
                      interval_seconds=args.interval)


def cmd_changelog(args) -> None:
    from qdrant_client import QdrantClient
    settings = get_settings()
    client = (QdrantClient(path="./qdrant_storage") if settings.qdrant_embedded
              else QdrantClient(url=settings.qdrant_url))
    store = ChangelogStore(client=client)
    store.ensure_collection()
    entries = store.list_recent(limit=args.limit)
    if not entries:
        print("Changelog is empty.")
        return
    for e in entries:
        print(f"[{e.detected_at}] ({e.form_type}) {e.title}\n    {e.summary}\n    {e.url or ''}")
```
Register them in `main()` (after the `ask` subparser block, before `args = parser.parse_args()`):
```python
    p_mon = sub.add_parser("monitor")
    p_mon.add_argument("--once", action="store_true", help="run a single poll and exit")
    p_mon.add_argument("--query", default="insider trading policy")
    p_mon.add_argument("--forms", default="8-K,10-K")
    p_mon.add_argument("--limit", type=int, default=10)
    p_mon.add_argument("--interval", type=int, default=3600, help="seconds between polls")
    p_mon.set_defaults(func=cmd_monitor)

    p_cl = sub.add_parser("changelog")
    p_cl.add_argument("--limit", type=int, default=20)
    p_cl.set_defaults(func=cmd_changelog)
```

- [ ] **Step 6: Add the gated live test `tests/test_monitor_live.py`**

```python
import pytest


@pytest.mark.live
def test_monitor_poll_live():
    """Requires Ollama (bge-m3 + gpt-oss) + network to SEC. Uses embedded Qdrant."""
    from qdrant_client import QdrantClient
    from regintel.config import Settings
    from regintel.monitoring.scheduler import build_default_monitor

    s = Settings(_env_file=None)
    monitor = build_default_monitor(s, client=QdrantClient(":memory:"))
    first = monitor.poll("insider trading policy", forms=["8-K"], limit=2)
    assert isinstance(first, list)
    # Cold start: at least one filing detected and recorded.
    assert len(first) >= 1
    assert all(e.summary for e in first)
    # Second poll: everything already seen -> no new entries.
    second = monitor.poll("insider trading policy", forms=["8-K"], limit=2)
    assert second == []
```

- [ ] **Step 7: Full suite + lint + CLI smoke**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all non-live tests PASS; ruff clean. Then `uv run python -m regintel.cli monitor --help` and `uv run python -m regintel.cli changelog --help` show usage.

- [ ] **Step 8: Commit**

```bash
git add src/regintel/monitoring/scheduler.py src/regintel/cli.py tests/test_scheduler.py tests/test_monitor_live.py
git commit -m "feat: add monitor scheduler + monitor/changelog CLI commands"
```

- [ ] **Step 9 (controller): live smoke**

Run (requires Ollama + network): `uv run pytest tests/test_monitor_live.py -m live -v`
Expected: PASS — cold-start poll records ≥1 changelog entry; second poll returns 0.

---

## Self-Review (completed by planner)

**Spec coverage:**
- apscheduler dep → Task 1 ✅
- ChangelogEntry → Task 1 ✅
- ChangelogStore (ensure/point_id/is_seen/record/list_recent) → Task 2 ✅
- MonitorAgent poll: search → filter unseen → fetch → ingest corpus → summarize → embed → record → Task 3 ✅
- Skip no-url / skip fetch-failure / skip already-seen → Task 3 tests ✅
- now_utc_iso → Task 3 ✅
- Scheduler (make_poll_job swallows errors, run_scheduler immediate+interval) → Task 4 ✅
- build_default_monitor wiring (shared client for corpus+changelog) → Task 4 ✅
- CLI monitor (--once / interval) + changelog → Task 4 ✅
- Live test (cold start ≥1, second poll 0) → Task 4 ✅
- Cold-start behavior → Task 3 (empty changelog → all new) + live test ✅

**Placeholder scan:** none — all code complete.

**Type consistency:** `ChangelogEntry(accession_no, title, form_type, filed_date, url, summary, detected_at)` constructed identically in Task 2 tests, Task 3 agent, and reconstructed via `ChangelogEntry(**payload)` in Task 2 store (payload = `asdict(entry)` → keys match exactly). `MonitorAgent(sec_client=, corpus_store=, dense=, sparse=, changelog_store=, provider=, summary_model=)` keyword set identical in Task 3 tests and Task 4 `build_default_monitor`. `ChangelogStore(client=, collection=)` consistent. `make_poll_job(monitor, *, query, forms, limit)` matches Task 4 test and `run_scheduler`. `DocInput`/`ingest_documents`/`SECFiling`/`OllamaEmbedder.embed_one` match the reused-signatures list. `SECFiling` positional order in `test_poll_skips_filing_without_url` (`SECFiling("a","No URL","8-K","2026-05-01", cik=None, doc_url=None)`) matches the dataclass field order (accession_no, title, form_type, filed_date, cik, doc_url).

**Note:** `list_recent` scrolls up to 10_000 points then sorts in memory — fine for a demo-scale changelog; documented as a known simplification.
```
