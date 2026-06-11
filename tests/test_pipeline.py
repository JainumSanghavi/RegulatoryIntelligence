from regintel.ingest.pipeline import DocInput, ingest_documents
from regintel.embeddings.sparse import SparseVec


class _FakeDense:
    def embed(self, texts):
        return [[0.1] * 1024 for _ in texts]


class _FakeSparse:
    def encode(self, texts):
        return [SparseVec([1], [1.0]) for _ in texts]


class _FakeStore:
    def __init__(self):
        self.records = []
    def ensure_collection(self):
        self.ensured = True
    def upsert(self, records):
        self.records.extend(records)


def test_ingest_chunks_embeds_upserts():
    store = _FakeStore()
    docs = [
        DocInput(doc_id="d1", title="T", text="word " * 1000, source="sec",
                 jurisdiction="US-SEC", doc_type="filing", form_type="10-K"),
    ]
    n = ingest_documents(docs, store=store, dense=_FakeDense(), sparse=_FakeSparse(),
                         chunk_tokens=100, overlap_tokens=20)
    assert store.ensured is True
    assert n == len(store.records) > 1
    rec = store.records[0]
    assert rec.payload.source == "sec"
    assert rec.payload.form_type == "10-K"
    assert len(rec.dense) == 1024
    assert rec.sparse_indices == [1]
