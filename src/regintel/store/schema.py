import uuid
from dataclasses import asdict, dataclass

COLLECTION = "corpus"
CHANGELOG_COLLECTION = "regulation_changelog"  # reserved for Phase 4
DENSE_VEC = "dense"
SPARSE_VEC = "sparse"
DENSE_DIM = 1024  # bge-m3

_NAMESPACE = uuid.UUID("11111111-1111-1111-1111-111111111111")


def point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic UUID5 so re-ingest updates rather than duplicates."""
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{chunk_index}"))


@dataclass
class ChunkPayload:
    doc_id: str
    chunk_index: int
    text: str
    source: str
    jurisdiction: str
    doc_type: str
    title: str
    url: str | None = None
    regulation_id: str | None = None
    form_type: str | None = None
    accession_no: str | None = None
    effective_date: str | None = None
    filed_date: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)
