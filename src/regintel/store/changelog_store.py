import uuid
from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from regintel.store.schema import CHANGELOG_COLLECTION, DENSE_DIM
from regintel.types import ChangelogEntry

_NAMESPACE = uuid.UUID("22222222-2222-2222-2222-222222222222")


class ChangelogStore:
    """Records detected SEC filings in the regulation_changelog collection."""

    def __init__(self, client: QdrantClient, collection: str = CHANGELOG_COLLECTION) -> None:
        self._client = client
        self._collection = collection

    @staticmethod
    def point_id(accession_no: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, accession_no))

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE),
        )

    def is_seen(self, accession_no: str) -> bool:
        got = self._client.retrieve(self._collection, ids=[self.point_id(accession_no)])
        return len(got) > 0

    def record(self, entry: ChangelogEntry, vector: list[float]) -> None:
        self._client.upsert(
            self._collection,
            points=[qm.PointStruct(id=self.point_id(entry.accession_no),
                                   vector=vector, payload=asdict(entry))],
        )

    def list_recent(self, limit: int = 20) -> list[ChangelogEntry]:
        points, _ = self._client.scroll(self._collection, limit=10_000,
                                        with_payload=True, with_vectors=False)
        entries = [ChangelogEntry(**p.payload) for p in points]
        entries.sort(key=lambda e: e.detected_at, reverse=True)
        return entries[:limit]
