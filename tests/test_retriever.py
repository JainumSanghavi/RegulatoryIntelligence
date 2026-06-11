from regintel.agents.retriever import RetrieverAgent
from regintel.embeddings.sparse import SparseVec
from regintel.types import RetrievalFilters, RetrievedChunk


class _FakeDense:
    def embed_one(self, text):
        return [0.1] * 1024


class _FakeSparse:
    def encode_query(self, text):
        return SparseVec([1], [1.0])


class _FakeStore:
    def __init__(self, results):
        self._results = results
        self.last_filters = None
    def hybrid_search(self, *, dense, sparse_indices, sparse_values, filters, limit, **kw):
        self.last_filters = filters
        return self._results


class _FakeProvider:
    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        return {"ranking": [{"index": 0, "rationale": "best"}]}


def test_retrieve_runs_full_pipeline():
    candidates = [
        RetrievedChunk("d1", 0, "relevant", 0.5, {"doc_id": "d1"}),
        RetrievedChunk("d2", 0, "less", 0.4, {"doc_id": "d2"}),
    ]
    agent = RetrieverAgent(
        store=_FakeStore(candidates), dense=_FakeDense(), sparse=_FakeSparse(),
        provider=_FakeProvider(), rerank_model="m", top_k=1,
    )
    out = agent.retrieve("insider trading", filters=RetrievalFilters(jurisdiction="US-SEC"))
    assert len(out) == 1
    assert out[0].doc_id == "d1"
    assert out[0].rerank_rationale == "best"


def test_retrieve_passes_filters_to_store():
    store = _FakeStore([])
    agent = RetrieverAgent(store=store, dense=_FakeDense(), sparse=_FakeSparse(),
                           provider=_FakeProvider(), rerank_model="m")
    agent.retrieve("q", filters=RetrievalFilters(source="internal"))
    assert store.last_filters.source == "internal"
