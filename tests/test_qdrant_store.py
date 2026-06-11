from qdrant_client import QdrantClient

from regintel.store.qdrant_store import QdrantStore, ChunkRecord
from regintel.store.schema import ChunkPayload


def _record(doc_id, idx, dense, sparse_idx, sparse_val, **payload_kw):
    payload = ChunkPayload(
        doc_id=doc_id, chunk_index=idx, text=f"text {idx}",
        source=payload_kw.get("source", "sec"),
        jurisdiction=payload_kw.get("jurisdiction", "US-SEC"),
        doc_type=payload_kw.get("doc_type", "filing"),
        title="T",
    )
    return ChunkRecord(
        payload=payload, dense=dense,
        sparse_indices=sparse_idx, sparse_values=sparse_val,
    )


def test_ensure_collection_and_upsert_count():
    store = QdrantStore(client=QdrantClient(":memory:"))
    store.ensure_collection()
    recs = [
        _record("d1", 0, [0.1] * 1024, [1, 5], [0.7, 0.3]),
        _record("d1", 1, [0.2] * 1024, [2, 5], [0.6, 0.4]),
    ]
    store.upsert(recs)
    assert store.count() == 2
    store.upsert(recs)
    assert store.count() == 2


def test_hybrid_search_filters_by_jurisdiction():
    store = QdrantStore(client=QdrantClient(":memory:"))
    store.ensure_collection()
    store.upsert([
        _record("sec1", 0, [0.9] + [0.0] * 1023, [1], [1.0], jurisdiction="US-SEC"),
        _record("int1", 0, [0.9] + [0.0] * 1023, [1], [1.0],
                jurisdiction="internal", source="internal", doc_type="policy"),
    ])
    from regintel.types import RetrievalFilters
    results = store.hybrid_search(
        dense=[0.9] + [0.0] * 1023,
        sparse_indices=[1], sparse_values=[1.0],
        filters=RetrievalFilters(jurisdiction="US-SEC"),
        limit=10,
    )
    assert len(results) == 1
    assert results[0].payload["jurisdiction"] == "US-SEC"
