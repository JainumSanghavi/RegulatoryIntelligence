from dataclasses import dataclass, field
from typing import Any, Literal

Source = Literal["sec", "internal"]
DocType = Literal["filing", "policy", "sop", "contract"]


@dataclass
class RetrievalFilters:
    """Payload filters for retrieval. None fields are ignored."""
    jurisdiction: str | None = None
    doc_type: str | None = None
    source: str | None = None
    date_from: str | None = None  # ISO date, filters filed_date >=
    date_to: str | None = None    # ISO date, filters filed_date <=

    def as_payload_conditions(self) -> dict[str, str]:
        """Equality conditions only (date range handled separately by the store)."""
        out: dict[str, str] = {}
        for key in ("jurisdiction", "doc_type", "source"):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        return out


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_index: int
    text: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    rerank_rationale: str | None = None
