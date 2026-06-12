from regintel.types import EvalScores, QueryType, Report


def test_evalscores_defaults_and_fields():
    s = EvalScores(faithfulness=0.8, citation_coverage=0.9, conflicts=[], flagged=False, notes="ok")
    assert s.faithfulness == 0.8
    assert s.conflicts == []
    assert s.flagged is False


def test_report_has_eval_field_defaulting_none():
    r = Report(query_type=QueryType.LOOKUP, answer="hi")
    assert r.eval is None
