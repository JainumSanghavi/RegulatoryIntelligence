from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from regintel.store.schema import (
    COLLECTION, DENSE_DIM, DENSE_VEC, SPARSE_VEC, ChunkPayload, point_id,
)
from regintel.types import RetrievalFilters, RetrievedChunk


@dataclass
class ChunkRecord:
    payload: ChunkPayload
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str = COLLECTION) -> None:
        self._client = client
        self._collection = collection

    @classmethod
    def from_settings(cls, settings) -> "QdrantStore":
        if settings.qdrant_embedded:
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(url=settings.qdrant_url)
        return cls(client)

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={DENSE_VEC: qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE)},
            sparse_vectors_config={SPARSE_VEC: qm.SparseVectorParams()},
        )

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count

    def upsert(self, records: list[ChunkRecord]) -> None:
        points = []
        for r in records:
            points.append(
                qm.PointStruct(
                    id=point_id(r.payload.doc_id, r.payload.chunk_index),
                    vector={
                        DENSE_VEC: r.dense,
                        SPARSE_VEC: qm.SparseVector(indices=r.sparse_indices, values=r.sparse_values),
                    },
                    payload=r.payload.as_dict(),
                )
            )
        self._client.upsert(self._collection, points=points)

    def _build_filter(self, filters: RetrievalFilters) -> qm.Filter | None:
        must: list[qm.FieldCondition] = []
        for key, val in filters.as_payload_conditions().items():
            must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=val)))
        if filters.date_from or filters.date_to:
            rng = qm.DatetimeRange(gte=filters.date_from, lte=filters.date_to)
            must.append(qm.FieldCondition(key="filed_date", range=rng))
        return qm.Filter(must=must) if must else None

    def hybrid_search(
        self,
        *,
        dense: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        filters: RetrievalFilters | None = None,
        limit: int = 20,
        prefetch_limit: int = 50,
    ) -> list[RetrievedChunk]:
        qfilter = self._build_filter(filters) if filters else None
        prefetch = [
            qm.Prefetch(query=dense, using=DENSE_VEC, limit=prefetch_limit, filter=qfilter),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse_indices, values=sparse_values),
                using=SPARSE_VEC, limit=prefetch_limit, filter=qfilter,
            ),
        ]
        resp = self._client.query_points(
            self._collection,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        out: list[RetrievedChunk] = []
        for p in resp.points:
            payload = p.payload or {}
            out.append(
                RetrievedChunk(
                    doc_id=payload.get("doc_id", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    text=payload.get("text", ""),
                    score=p.score,
                    payload=payload,
                )
            )
        return out
