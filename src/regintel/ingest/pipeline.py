from dataclasses import dataclass

from regintel.ingest.chunker import chunk_text
from regintel.store.qdrant_store import ChunkRecord
from regintel.store.schema import ChunkPayload


@dataclass
class DocInput:
    doc_id: str
    title: str
    text: str
    source: str
    jurisdiction: str
    doc_type: str
    url: str | None = None
    regulation_id: str | None = None
    form_type: str | None = None
    accession_no: str | None = None
    effective_date: str | None = None
    filed_date: str | None = None


def ingest_documents(
    docs: list[DocInput],
    *,
    store,
    dense,
    sparse,
    chunk_tokens: int = 800,
    overlap_tokens: int = 150,
) -> int:
    store.ensure_collection()
    records: list[ChunkRecord] = []
    for doc in docs:
        chunks = chunk_text(
            doc.text, doc_id=doc.doc_id,
            chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
        )
        if not chunks:
            continue
        texts = [c.text for c in chunks]
        dense_vecs = dense.embed(texts)
        sparse_vecs = sparse.encode(texts)
        for chunk, dvec, svec in zip(chunks, dense_vecs, sparse_vecs, strict=True):
            payload = ChunkPayload(
                doc_id=doc.doc_id, chunk_index=chunk.chunk_index, text=chunk.text,
                source=doc.source, jurisdiction=doc.jurisdiction, doc_type=doc.doc_type,
                title=doc.title, url=doc.url, regulation_id=doc.regulation_id,
                form_type=doc.form_type, accession_no=doc.accession_no,
                effective_date=doc.effective_date, filed_date=doc.filed_date,
            )
            records.append(
                ChunkRecord(payload=payload, dense=dvec,
                            sparse_indices=svec.indices, sparse_values=svec.values)
            )
    if records:
        store.upsert(records)
    return len(records)
