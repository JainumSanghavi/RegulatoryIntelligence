from regintel.agents.orchestrator import Orchestrator
from regintel.types import QueryType


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        self.calls.append(messages)
        return self._payload


def test_classify_returns_enum():
    p = _FakeProvider({"query_type": "impact", "reasoning": "a rule changed"})
    assert Orchestrator(p, model="m").classify("How does the new rule affect us?") is QueryType.IMPACT


def test_classify_gap_check():
    p = _FakeProvider({"query_type": "gap_check", "reasoning": "compliance check"})
    assert Orchestrator(p, model="m").classify("Does our policy comply?") is QueryType.GAP_CHECK


def test_classify_bad_value_falls_back_to_gap_check():
    p = _FakeProvider({"query_type": "nonsense", "reasoning": "x"})
    assert Orchestrator(p, model="m").classify("q") is QueryType.GAP_CHECK


def test_classify_missing_key_falls_back():
    p = _FakeProvider({"reasoning": "x"})
    assert Orchestrator(p, model="m").classify("q") is QueryType.GAP_CHECK
