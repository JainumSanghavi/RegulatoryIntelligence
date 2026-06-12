from regintel.orchestration.graph import build_graph
from regintel.state import new_state
from regintel.types import Finding, QueryType, Report, RetrievedChunk


class _FakeRetriever:
    def __init__(self, regs, internal):
        self._regs = regs
        self._internal = internal
        self.calls = []

    def retrieve(self, query, *, filters=None):
        src = getattr(filters, "source", None)
        juris = getattr(filters, "jurisdiction", None)
        self.calls.append(juris or src)
        return self._internal if src == "internal" else self._regs


class _FakeOrch:
    def __init__(self, qt):
        self.qt = qt

    def classify(self, query):
        return self.qt


class _RecordingAnalyst:
    def __init__(self, findings):
        self.findings = findings
        self.called = False

    def analyze(self, query, regs, internal):
        self.called = True
        return self.findings


class _RecordingAssessor:
    def __init__(self):
        self.called = False

    def assess(self, findings, internal):
        self.called = True
        return []


class _NoopEvaluator:
    def evaluate(self, query, report):
        from regintel.types import EvalScores
        return EvalScores(1.0, 1.0, [], flagged=False, notes="ok")


class _FakeReporter:
    def report(self, query, query_type, findings, impacts, regs, internal):
        return Report(query_type=query_type, answer="report", findings=findings)


def _chunk():
    return RetrievedChunk(doc_id="d", chunk_index=0, text="t", score=0.5,
                          payload={"title": "T", "source": "sec", "url": None})


def _build(qt, regs, findings):
    analyst = _RecordingAnalyst(findings)
    assessor = _RecordingAssessor()
    graph = build_graph(
        retriever=_FakeRetriever(regs, [_chunk()]),
        orchestrator=_FakeOrch(qt),
        analyst=analyst,
        assessor=assessor,
        reporter=_FakeReporter(),
        evaluator=_NoopEvaluator(),
    )
    return graph, analyst, assessor


def test_lookup_skips_analyst_and_assessor():
    graph, analyst, assessor = _build(QueryType.LOOKUP, [_chunk()], [])
    final = graph.invoke(new_state("q"))
    assert analyst.called is False
    assert assessor.called is False
    assert isinstance(final["report"], Report)


def test_empty_regulations_short_circuits_to_report():
    graph, analyst, assessor = _build(QueryType.GAP_CHECK, [], [])
    graph.invoke(new_state("q"))
    assert analyst.called is False
    assert assessor.called is False


def test_gap_check_with_findings_runs_assessor():
    finding = Finding(topic="t", requirement="r", internal_status="absent",
                      gap=True, explanation="e")
    graph, analyst, assessor = _build(QueryType.GAP_CHECK, [_chunk()], [finding])
    graph.invoke(new_state("q"))
    assert analyst.called is True
    assert assessor.called is True


def test_gap_check_no_findings_skips_assessor():
    graph, analyst, assessor = _build(QueryType.GAP_CHECK, [_chunk()], [])
    graph.invoke(new_state("q"))
    assert analyst.called is True
    assert assessor.called is False
