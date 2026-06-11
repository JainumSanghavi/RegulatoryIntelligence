import pytest

from regintel.embeddings.sparse import BM25Encoder, SparseVec


class _FakeEmbedding:
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values


class _FakeModel:
    def embed(self, texts):
        for i, _ in enumerate(texts):
            yield _FakeEmbedding([i, i + 1], [0.5, 0.5])

    def query_embed(self, text):
        yield _FakeEmbedding([0], [1.0])


def test_encode_documents_with_injected_model():
    enc = BM25Encoder(model=_FakeModel())
    out = enc.encode(["doc one", "doc two"])
    assert isinstance(out[0], SparseVec)
    assert out[0].indices == [0, 1]
    assert out[1].indices == [1, 2]


def test_encode_query_with_injected_model():
    enc = BM25Encoder(model=_FakeModel())
    q = enc.encode_query("find this")
    assert q.indices == [0]
    assert q.values == [1.0]


@pytest.mark.live
def test_real_bm25_loads():
    enc = BM25Encoder()  # downloads Qdrant/bm25
    out = enc.encode(["insider trading blackout window policy"])
    assert len(out[0].indices) > 0
