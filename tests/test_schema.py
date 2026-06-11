from regintel.store.schema import (
    COLLECTION, DENSE_DIM, DENSE_VEC, SPARSE_VEC, ChunkPayload, point_id,
)


def test_constants():
    assert COLLECTION == "corpus"
    assert DENSE_DIM == 1024
    assert DENSE_VEC == "dense"
    assert SPARSE_VEC == "sparse"


def test_point_id_deterministic():
    a = point_id("docA", 3)
    b = point_id("docA", 3)
    c = point_id("docA", 4)
    assert a == b
    assert a != c


def test_chunk_payload_to_dict():
    p = ChunkPayload(
        doc_id="d1", chunk_index=0, text="t", source="sec",
        jurisdiction="US-SEC", doc_type="filing", title="10-K",
        form_type="10-K", accession_no="000", filed_date="2026-01-01",
    )
    d = p.as_dict()
    assert d["source"] == "sec"
    assert d["form_type"] == "10-K"
    assert d["url"] is None
