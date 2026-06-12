from regintel.orchestration.graph import build_graph
from regintel.state import new_state
from regintel.types import EvalScores, QueryType, Report, RetrievedChunk


class _FakeRetriever:
    def retrieve(self, query, *, filters=None):
        return [RetrievedChunk(doc_id="d", chunk_index=0, text="t", score=0.5,
                               payload={"title": "T", "source": "sec", "url": None})]


class _FakeOrch:
    def classify(self, query):
        return QueryType.LOOKUP


class _FakeReporter:
    def report(self, query, query_type, findings, impacts, regs, internal):
        return Report(query_type=query_type, answer="answer [0]")


class _FakeEvaluator:
    def __init__(self):
        self.called_with = None

    def evaluate(self, query, report):
        self.called_with = report
        return EvalScores(0.5, 0.5, [], flagged=True, notes="low faithfulness or conflicts detected")


def _graph(evaluator):
    return build_graph(
        retriever=_FakeRetriever(), orchestrator=_FakeOrch(),
        analyst=None, assessor=None, reporter=_FakeReporter(), evaluator=evaluator,
    )


def test_evaluate_runs_after_report_and_attaches_scores():
    ev = _FakeEvaluator()
    final = _graph(ev).invoke(new_state("q"))
    assert ev.called_with is not None
    assert isinstance(final["eval_scores"], EvalScores)
    assert final["report"].eval is final["eval_scores"]


def test_flagged_report_gets_warning():
    final = _graph(_FakeEvaluator()).invoke(new_state("q"))
    assert any("low-confidence" in w for w in final["report"].warnings)
