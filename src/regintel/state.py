from typing import Any, TypedDict

from regintel.types import RetrievalFilters, RetrievedChunk


class AgentState(TypedDict, total=False):
    """Full LangGraph state. Phases 0+1 populate only query/filters/retrieved/errors."""
    query: str
    sub_questions: list[str]
    filters: RetrievalFilters
    retrieved: list[RetrievedChunk]
    analyst_findings: list[dict[str, Any]]      # Phase 2
    impact_assessments: list[dict[str, Any]]    # Phase 2
    report: dict[str, Any] | None               # Phase 2
    eval_scores: dict[str, Any] | None          # Phase 3
    errors: list[str]
    messages: list[dict[str, Any]]


def new_state(query: str) -> AgentState:
    return AgentState(
        query=query,
        sub_questions=[],
        filters=RetrievalFilters(),
        retrieved=[],
        errors=[],
        messages=[],
    )
