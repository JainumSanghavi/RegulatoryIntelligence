from regintel.agents.evaluator import Evaluator, FAITHFULNESS_THRESHOLD
from regintel.types import Citation, EvalScores, QueryType, Report


def test_evalscores_defaults_and_fields():
    s = EvalScores(faithfulness=0.8, citation_coverage=0.9, conflicts=[], flagged=False, notes="ok")
    assert s.faithfulness == 0.8
    assert s.conflicts == []
    assert s.flagged is False


def test_report_has_eval_field_defaulting_none():
    r = Report(query_type=QueryType.LOOKUP, answer="hi")
    assert r.eval is None



class _SchemaKeyedProvider:
    """Returns a payload based on which schema is requested; records call count."""
    def __init__(self, claims=None, conflicts=None, raise_on_claims=False):
        self._claims = claims if claims is not None else []
        self._conflicts = conflicts if conflicts is not None else []
        self.raise_on_claims = raise_on_claims
        self.calls = 0

    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        self.calls += 1
        props = schema.get("properties", {})
        if "claims" in props:
            if self.raise_on_claims:
                from regintel.llm.base import LLMError
                raise LLMError("boom")
            return {"claims": self._claims}
        if "conflicts" in props:
            return {"conflicts": self._conflicts}
        raise AssertionError("unexpected schema")


def _report(answer="The policy lacks a blackout window [0].", with_citation=True):
    cits = [Citation("d", 0, "SEC Rule", "sec", "http://x", "blackout windows required")] if with_citation else []
    return Report(query_type=QueryType.GAP_CHECK, answer=answer, citations=cits)


def test_faithfulness_and_coverage_math():
    claims = [
        {"claim": "a", "supported": True, "has_citation": True},
        {"claim": "b", "supported": True, "has_citation": False},
        {"claim": "c", "supported": False, "has_citation": True},
    ]
    ev = Evaluator(_SchemaKeyedProvider(claims=claims), model="m")
    scores = ev.evaluate("q", _report())
    assert round(scores.faithfulness, 2) == 0.67
    assert round(scores.citation_coverage, 2) == 0.67
    assert scores.flagged is True


def test_high_faithfulness_not_flagged():
    claims = [{"claim": "a", "supported": True, "has_citation": True},
              {"claim": "b", "supported": True, "has_citation": True}]
    ev = Evaluator(_SchemaKeyedProvider(claims=claims), model="m")
    scores = ev.evaluate("q", _report())
    assert scores.faithfulness == 1.0
    assert scores.flagged is False


def test_conflicts_force_flag_even_if_faithful():
    claims = [{"claim": "a", "supported": True, "has_citation": True}]
    ev = Evaluator(_SchemaKeyedProvider(claims=claims,
                                        conflicts=[{"description": "passage 0 contradicts passage 1"}]),
                   model="m")
    scores = ev.evaluate("q", _report())
    assert scores.faithfulness == 1.0
    assert scores.conflicts == ["passage 0 contradicts passage 1"]
    assert scores.flagged is True


def test_short_circuit_no_citations_skips_llm():
    provider = _SchemaKeyedProvider()
    ev = Evaluator(provider, model="m")
    scores = ev.evaluate("q", _report(with_citation=False))
    assert provider.calls == 0
    assert scores.flagged is False
    assert scores.faithfulness == 1.0


def test_short_circuit_no_evidence_answer():
    provider = _SchemaKeyedProvider()
    ev = Evaluator(provider, model="m")
    rep = Report(query_type=QueryType.GAP_CHECK,
                 answer="No relevant regulations found for this question.",
                 citations=[Citation("d", 0, "t", "sec", None, "q")])
    scores = ev.evaluate("q", rep)
    assert provider.calls == 0
    assert scores.flagged is False


def test_fail_safe_on_provider_error():
    ev = Evaluator(_SchemaKeyedProvider(raise_on_claims=True), model="m")
    scores = ev.evaluate("q", _report())
    assert scores.flagged is True
    assert scores.faithfulness == 0.0
    assert "failed" in scores.notes.lower()


def test_threshold_constant():
    assert FAITHFULNESS_THRESHOLD == 0.7
