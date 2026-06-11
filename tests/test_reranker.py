from regintel.rerank.llm_reranker import rerank
from regintel.types import RetrievedChunk


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload
    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        return self._payload


def _chunks():
    return [
        RetrievedChunk("d1", 0, "irrelevant text", 0.5, {"doc_id": "d1"}),
        RetrievedChunk("d2", 0, "highly relevant text", 0.4, {"doc_id": "d2"}),
    ]


def test_rerank_reorders_and_limits():
    provider = _FakeProvider({"ranking": [
        {"index": 1, "rationale": "directly answers"},
        {"index": 0, "rationale": "tangential"},
    ]})
    out = rerank("q", _chunks(), provider=provider, model="m", top_k=1)
    assert len(out) == 1
    assert out[0].doc_id == "d2"
    assert out[0].rerank_rationale == "directly answers"


def test_rerank_ignores_out_of_range_indices():
    provider = _FakeProvider({"ranking": [{"index": 99, "rationale": "x"}, {"index": 0, "rationale": "y"}]})
    out = rerank("q", _chunks(), provider=provider, model="m", top_k=5)
    assert [c.doc_id for c in out] == ["d1"]
