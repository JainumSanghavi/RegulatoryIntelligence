from regintel.state import AgentState
from regintel.types import EvalScores, QueryType, Report, RetrievalFilters


def _append_error(state: AgentState, msg: str) -> list[str]:
    return list(state.get("errors", [])) + [msg]


def make_classify_node(orchestrator):
    def node(state: AgentState) -> dict:
        try:
            return {"query_type": orchestrator.classify(state["query"])}
        except Exception as exc:  # noqa: BLE001
            return {"query_type": QueryType.GAP_CHECK,
                    "errors": _append_error(state, f"classify: {exc}")}
    return node


def make_retrieve_regulations_node(retriever):
    def node(state: AgentState) -> dict:
        try:
            chunks = retriever.retrieve(state["query"],
                                        filters=RetrievalFilters(jurisdiction="US-SEC"))
            return {"retrieved": chunks}
        except Exception as exc:  # noqa: BLE001
            return {"retrieved": [], "errors": _append_error(state, f"retrieve_regulations: {exc}")}
    return node


def make_retrieve_internal_node(retriever):
    def node(state: AgentState) -> dict:
        try:
            chunks = retriever.retrieve(state["query"],
                                        filters=RetrievalFilters(source="internal"))
            return {"internal": chunks}
        except Exception as exc:  # noqa: BLE001
            return {"internal": [], "errors": _append_error(state, f"retrieve_internal: {exc}")}
    return node


def make_analyze_node(analyst):
    def node(state: AgentState) -> dict:
        try:
            findings = analyst.analyze(state["query"], state.get("retrieved", []),
                                       state.get("internal", []))
            return {"analyst_findings": findings}
        except Exception as exc:  # noqa: BLE001
            return {"analyst_findings": [], "errors": _append_error(state, f"analyze: {exc}")}
    return node


def make_assess_node(assessor):
    def node(state: AgentState) -> dict:
        try:
            impacts = assessor.assess(state.get("analyst_findings", []),
                                      state.get("internal", []))
            return {"impact_assessments": impacts}
        except Exception as exc:  # noqa: BLE001
            return {"impact_assessments": [], "errors": _append_error(state, f"assess: {exc}")}
    return node


def make_report_node(reporter):
    def node(state: AgentState) -> dict:
        qt = state.get("query_type", QueryType.GAP_CHECK)
        try:
            report = reporter.report(
                state["query"], qt, state.get("analyst_findings", []),
                state.get("impact_assessments", []), state.get("retrieved", []),
                state.get("internal", []),
            )
        except Exception as exc:  # noqa: BLE001
            report = Report(query_type=qt, answer="Unable to generate a report.",
                            warnings=[f"report: {exc}"])
        report.warnings = list(report.warnings) + list(state.get("errors", []))
        return {"report": report}
    return node


def make_evaluate_node(evaluator):
    def node(state: AgentState) -> dict:
        report = state.get("report")
        if report is None:
            return {}
        try:
            scores = evaluator.evaluate(state.get("query", ""), report)
        except Exception as exc:  # noqa: BLE001
            scores = EvalScores(0.0, 0.0, [], flagged=True, notes=f"evaluation failed: {exc}")
        report.eval = scores
        if scores.flagged:
            report.warnings = list(report.warnings) + [f"low-confidence: {scores.notes}"]
        return {"eval_scores": scores, "report": report}
    return node


def route_after_regulations(state: AgentState) -> str:
    if not state.get("retrieved"):
        return "report"
    if state.get("query_type") == QueryType.LOOKUP:
        return "report"
    return "retrieve_internal"


def route_after_analyze(state: AgentState) -> str:
    return "assess" if state.get("analyst_findings") else "report"
