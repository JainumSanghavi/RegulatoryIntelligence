from regintel.agents.analyst import Analyst
from regintel.types import RetrievedChunk


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload

    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        return self._payload


def _chunk(doc_id, idx, text, title, source):
    return RetrievedChunk(doc_id=doc_id, chunk_index=idx, text=text, score=0.5,
                          payload={"title": title, "source": source, "url": None})


def test_analyze_resolves_citations_from_indices():
    regs = [_chunk("sec1", 0, "Blackout windows required.", "SEC 8-K", "sec")]
    internal = [_chunk("pol1", 0, "No blackout clause.", "Insider Policy", "internal")]
    payload = {"findings": [{
        "topic": "blackout windows", "requirement": "must define blackout window",
        "internal_status": "absent", "gap": True, "explanation": "policy lacks it",
        "regulation_refs": [0], "internal_refs": [0],
    }]}
    findings = Analyst(_FakeProvider(payload), model="m").analyze("q", regs, internal)
    assert len(findings) == 1
    f = findings[0]
    assert f.gap is True and f.topic == "blackout windows"
    assert {c.source for c in f.citations} == {"sec", "internal"}
    assert any(c.title == "SEC 8-K" for c in f.citations)


def test_analyze_drops_out_of_range_refs():
    regs = [_chunk("sec1", 0, "text", "SEC", "sec")]
    payload = {"findings": [{
        "topic": "t", "requirement": "r", "internal_status": "absent",
        "gap": False, "explanation": "e", "regulation_refs": [5], "internal_refs": [],
    }]}
    findings = Analyst(_FakeProvider(payload), model="m").analyze("q", regs, [])
    assert findings[0].citations == []


def test_analyze_empty_regulations_returns_empty():
    findings = Analyst(_FakeProvider({"findings": []}), model="m").analyze("q", [], [])
    assert findings == []
