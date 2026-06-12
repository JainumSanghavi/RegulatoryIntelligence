from regintel.types import (
    Citation, Finding, Impact, QueryType, Report, RetrievedChunk, cite,
)
from regintel.state import new_state


def test_query_type_values():
    assert QueryType("lookup") is QueryType.LOOKUP
    assert QueryType.GAP_CHECK.value == "gap_check"
    assert QueryType.IMPACT.value == "impact"


def test_cite_from_chunk():
    chunk = RetrievedChunk(
        doc_id="d1", chunk_index=2, text="x" * 400, score=0.5,
        payload={"title": "Policy A", "source": "internal", "url": None},
    )
    c = cite(chunk)
    assert isinstance(c, Citation)
    assert c.doc_id == "d1" and c.chunk_index == 2
    assert c.title == "Policy A" and c.source == "internal"
    assert len(c.quote) <= 300


def test_finding_and_impact_defaults():
    f = Finding(topic="t", requirement="r", internal_status="absent", gap=True, explanation="e")
    assert f.citations == []
    im = Impact(topic="t", affected_policies=["Policy A"], severity="high", rationale="r")
    assert im.severity == "high"


def test_report_defaults():
    r = Report(query_type=QueryType.LOOKUP, answer="hello")
    assert r.citations == [] and r.findings == [] and r.impacts == [] and r.warnings == []


def test_new_state_has_phase2_slots():
    s = new_state("q")
    assert s["internal"] == []
    assert s["retrieved"] == []
