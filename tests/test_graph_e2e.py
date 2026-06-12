from regintel.orchestration.graph import build_graph, run_query
from regintel.types import QueryType, Report, RetrievedChunk


class _StubProvider:
    """Returns canned structured output keyed by the schema's required fields."""
    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        props = schema.get("properties", {})
        if "query_type" in props:
            return {"query_type": "gap_check", "reasoning": "compliance"}
        if "findings" in props:
            return {"findings": [{
                "topic": "blackout windows", "requirement": "define blackout",
                "internal_status": "absent", "gap": True, "explanation": "missing",
                "regulation_refs": [0], "internal_refs": [0],
            }]}
        if "impacts" in props:
            return {"impacts": [{"topic": "blackout windows",
                                 "affected_policies": ["Insider Policy"],
                                 "severity": "high", "rationale": "material"}]}
        if "claims" in props:
            return {"claims": [{"claim": "Gap in blackout windows", "supported": True, "has_citation": True}]}
        if "conflicts" in props:
            return {"conflicts": []}
        return {"answer": "Gap in blackout windows [0].", "cited_indices": [0]}


class _Retriever:
    def retrieve(self, query, *, filters=None):
        src = getattr(filters, "source", None)
        title = "Insider Policy" if src == "internal" else "SEC Rule"
        return [RetrievedChunk(doc_id=title, chunk_index=0, text="text", score=0.5,
                               payload={"title": title, "source": src or "sec", "url": None})]


def test_end_to_end_gap_check_report():
    from regintel.agents.analyst import Analyst
    from regintel.agents.evaluator import Evaluator
    from regintel.agents.impact_assessor import ImpactAssessor
    from regintel.agents.orchestrator import Orchestrator
    from regintel.agents.reporter import Reporter

    p = _StubProvider()
    graph = build_graph(
        retriever=_Retriever(),
        orchestrator=Orchestrator(p, "m"), analyst=Analyst(p, "m"),
        assessor=ImpactAssessor(p, "m"), reporter=Reporter(p, "m"),
        evaluator=Evaluator(p, "m"),
    )
    report = run_query("Does our insider policy comply?", graph=graph)
    assert isinstance(report, Report)
    assert report.query_type is QueryType.GAP_CHECK
    assert report.findings and report.findings[0].gap is True
    assert report.impacts and report.impacts[0].severity == "high"
    assert report.citations  # at least one resolved citation
    assert "blackout" in report.answer.lower()
    assert report.eval is not None
    assert 0.0 <= report.eval.faithfulness <= 1.0
    assert report.eval.faithfulness == 1.0
