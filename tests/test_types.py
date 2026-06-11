from regintel.types import RetrievalFilters, RetrievedChunk


def test_filters_to_dict_drops_none():
    f = RetrievalFilters(jurisdiction="US-SEC", source=None)
    assert f.as_payload_conditions() == {"jurisdiction": "US-SEC"}


def test_retrieved_chunk_roundtrip():
    c = RetrievedChunk(
        doc_id="d1", chunk_index=0, text="hello",
        score=0.9, payload={"source": "sec"},
    )
    assert c.doc_id == "d1"
    assert c.score == 0.9
    assert c.payload["source"] == "sec"
