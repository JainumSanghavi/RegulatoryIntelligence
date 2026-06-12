from dataclasses import dataclass, field
from enum import Enum
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


class QueryType(str, Enum):
    LOOKUP = "lookup"
    GAP_CHECK = "gap_check"
    IMPACT = "impact"


@dataclass
class Citation:
    doc_id: str
    chunk_index: int
    title: str
    source: str
    url: str | None
    quote: str


def cite(chunk: "RetrievedChunk", *, max_quote: int = 300) -> Citation:
    """Build a Citation from a retrieved chunk (quote truncated)."""
    p = chunk.payload or {}
    return Citation(
        doc_id=chunk.doc_id,
        chunk_index=chunk.chunk_index,
        title=p.get("title", ""),
        source=p.get("source", ""),
        url=p.get("url"),
        quote=chunk.text[:max_quote],
    )


@dataclass
class Finding:
    topic: str
    requirement: str
    internal_status: str
    gap: bool
    explanation: str
    citations: list[Citation] = field(default_factory=list)


Severity = Literal["low", "medium", "high", "critical"]


@dataclass
class Impact:
    topic: str
    affected_policies: list[str]
    severity: str
    rationale: str


@dataclass
class Report:
    query_type: QueryType
    answer: str
    citations: list[Citation] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    impacts: list[Impact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
