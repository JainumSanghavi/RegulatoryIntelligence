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
    assert len(corpus.records) >= 2
    assert set(changelog.seen) == {"a", "b"}
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
    assert sec.fetched == []


def test_poll_skips_filing_whose_fetch_fails():
    sec = _FakeSEC([_filing("a", "http://x/a"), _filing("b", "http://x/b")],
                   {"http://x/b": "body b"}, raise_for={"http://x/a"})
    changelog = _FakeChangelog()
    monitor = _monitor(sec, changelog)
    entries = monitor.poll("q", forms=["8-K"], limit=10)
    assert [e.accession_no for e in entries] == ["b"]
