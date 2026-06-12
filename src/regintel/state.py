from typing import Any, TypedDict

from regintel.types import (
    Finding, Impact, QueryType, Report, RetrievalFilters, RetrievedChunk,
)


class AgentState(TypedDict, total=False):
    """Full LangGraph state for the RegIntel pipeline."""
    query: str
    query_type: QueryType
    sub_questions: list[str]
    filters: RetrievalFilters
    retrieved: list[RetrievedChunk]      # regulatory (SEC) hits
    internal: list[RetrievedChunk]       # internal-doc hits
    analyst_findings: list[Finding]
    impact_assessments: list[Impact]
    report: Report | None
    eval_scores: dict[str, Any] | None   # Phase 3
    errors: list[str]
    messages: list[dict[str, Any]]


def new_state(query: str) -> AgentState:
    return AgentState(
        query=query,
        sub_questions=[],
        filters=RetrievalFilters(),
        retrieved=[],
        internal=[],
        analyst_findings=[],
        impact_assessments=[],
        errors=[],
        messages=[],
    )
