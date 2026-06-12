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


def test_index_serves_html():
    r = _client().get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Regulatory Intelligence" in r.text
    assert "/ask" in r.text
