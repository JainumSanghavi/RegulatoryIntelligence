from regintel.agents.reporter import Reporter
from regintel.types import Citation, Finding, QueryType, RetrievedChunk


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload
        self.last_user = None

    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        self.last_user = messages[-1].content
        return self._payload


def _chunk(doc_id, title):
    return RetrievedChunk(doc_id=doc_id, chunk_index=0, text="regulation text", score=0.5,
                          payload={"title": title, "source": "sec", "url": "http://x"})


def test_report_lookup_uses_regulation_chunks():
    provider = _FakeProvider({"answer": "Rule says X [0].", "cited_indices": [0]})
    regs = [_chunk("sec1", "SEC Rule")]
    rep = Reporter(provider, model="m").report("q", QueryType.LOOKUP, [], [], regs, [])
    assert rep.query_type is QueryType.LOOKUP
    assert "Rule says X" in rep.answer
    assert len(rep.citations) == 1 and rep.citations[0].title == "SEC Rule"


def test_report_keeps_only_cited_indices():
    provider = _FakeProvider({"answer": "Only first [0].", "cited_indices": [0]})
    regs = [_chunk("sec1", "First"), _chunk("sec2", "Second")]
    rep = Reporter(provider, model="m").report("q", QueryType.LOOKUP, [], [], regs, [])
    assert [c.title for c in rep.citations] == ["First"]


def test_report_no_evidence_short_circuits_without_llm():
    provider = _FakeProvider({"answer": "should not be used"})
    rep = Reporter(provider, model="m").report("q", QueryType.GAP_CHECK, [], [], [], [])
    assert "no relevant regulations" in rep.answer.lower()
    assert provider.last_user is None  # LLM not called


def test_report_includes_findings_and_impacts():
    provider = _FakeProvider({"answer": "Gap found [0].", "cited_indices": [0]})
    finding = Finding(topic="blackout", requirement="r", internal_status="absent",
                      gap=True, explanation="e",
                      citations=[Citation("d", 0, "Policy", "internal", None, "snippet")])
    rep = Reporter(provider, model="m").report("q", QueryType.GAP_CHECK, [finding], [], [], [])
    assert rep.findings == [finding]
    assert len(rep.citations) == 1
