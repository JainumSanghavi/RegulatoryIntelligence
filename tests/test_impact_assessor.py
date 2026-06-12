from regintel.agents.impact_assessor import ImpactAssessor
from regintel.types import Finding, RetrievedChunk


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload

    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        return self._payload


def _internal(title):
    return RetrievedChunk(doc_id=title, chunk_index=0, text="t", score=0.5,
                          payload={"title": title, "source": "internal", "url": None})


def _gap_finding(topic="blackout windows"):
    return Finding(topic=topic, requirement="r", internal_status="absent",
                   gap=True, explanation="e")


def test_assess_returns_impacts_and_validates_policies():
    payload = {"impacts": [{
        "topic": "blackout windows",
        "affected_policies": ["Insider Policy", "Nonexistent Policy"],
        "severity": "high", "rationale": "material gap",
    }]}
    internal = [_internal("Insider Policy")]
    impacts = ImpactAssessor(_FakeProvider(payload), model="m").assess([_gap_finding()], internal)
    assert len(impacts) == 1
    assert impacts[0].affected_policies == ["Insider Policy"]  # unknown dropped
    assert impacts[0].severity == "high"


def test_assess_coerces_bad_severity_to_medium():
    payload = {"impacts": [{"topic": "blackout windows", "affected_policies": [],
                            "severity": "catastrophic", "rationale": "r"}]}
    impacts = ImpactAssessor(_FakeProvider(payload), model="m").assess([_gap_finding()], [])
    assert impacts[0].severity == "medium"


def test_assess_no_gaps_returns_empty():
    non_gap = Finding(topic="t", requirement="r", internal_status="present",
                      gap=False, explanation="e")
    impacts = ImpactAssessor(_FakeProvider({"impacts": []}), model="m").assess([non_gap], [])
    assert impacts == []
