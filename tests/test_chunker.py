from regintel.ingest.chunker import chunk_text


def test_chunk_text_indices_sequential():
    text = "word " * 2000
    chunks = chunk_text(text, doc_id="d1", chunk_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_short_text_single_chunk():
    chunks = chunk_text("just a short bit", doc_id="d2")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
