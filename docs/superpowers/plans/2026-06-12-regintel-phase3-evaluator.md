# RegIntel Phase 3 Implementation Plan (Evaluator / Trust Layer)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an EvaluatorAgent that scores each report for faithfulness, citation coverage, and cross-chunk conflicts (custom LLM-as-judge, zero new deps), wired as a final `evaluate` node that flags low-confidence answers.

**Architecture:** A new `Evaluator` (frontier model) makes two `chat_structured` calls — claims (faithfulness + citation coverage) and conflicts — over the report's cited passages. An `evaluate` node runs after `report` in the LangGraph, attaches `EvalScores` to the `Report`, and appends a warning when flagged. Reuses the hardened `OllamaProvider`.

**Tech Stack:** existing `regintel` modules (OllamaProvider, LangGraph graph, router), pytest. No new dependencies.

**Conventions:** run via `uv run` (plain `uv run pytest`, no `--extra`); git author Jainum Sanghavi <sanghavi.h.j20@gmail.com>, NO Co-Authored-By trailer; commit per task with the exact message; if you must touch `pyproject.toml`/`uv.lock`, STOP and report.

---

## File structure

```
src/regintel/
  types.py                  # MODIFY: + EvalScores; Report.eval field
  state.py                  # MODIFY: eval_scores: EvalScores | None
  agents/evaluator.py       # CREATE: Evaluator
  orchestration/nodes.py    # MODIFY: + make_evaluate_node
  orchestration/graph.py    # MODIFY: evaluator param, evaluate node, report->evaluate->END
  cli.py                    # MODIFY: print Evaluation section
tests/
  test_evaluator.py         # CREATE
  test_graph_eval.py        # CREATE
  test_graph_e2e.py         # MODIFY: stub eval payloads + assert report.eval
  test_ask_live.py          # MODIFY: assert report.eval
```

---

## Task 1: Data contract (`EvalScores`, `Report.eval`, state)

**Files:** Modify `src/regintel/types.py`, `src/regintel/state.py`; Test `tests/test_evaluator.py` (types portion)

- [ ] **Step 1: Write the failing test**

`tests/test_evaluator.py`:
```python
from regintel.types import EvalScores, QueryType, Report


def test_evalscores_defaults_and_fields():
    s = EvalScores(faithfulness=0.8, citation_coverage=0.9, conflicts=[], flagged=False, notes="ok")
    assert s.faithfulness == 0.8
    assert s.conflicts == []
    assert s.flagged is False


def test_report_has_eval_field_defaulting_none():
    r = Report(query_type=QueryType.LOOKUP, answer="hi")
    assert r.eval is None
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: FAIL (ImportError: cannot import name 'EvalScores').

- [ ] **Step 3: Add `EvalScores` to `src/regintel/types.py`**

Append:
```python
@dataclass
class EvalScores:
    faithfulness: float        # 0..1 — fraction of answer claims supported by cited passages
    citation_coverage: float   # 0..1 — fraction of answer claims carrying a citation
    conflicts: list[str]       # cross-chunk contradictions (empty = none)
    flagged: bool              # below threshold OR conflicts present
    notes: str                 # brief judge rationale
```

- [ ] **Step 4: Add the `eval` field to `Report`**

In the existing `Report` dataclass, add as the last field:
```python
    eval: "EvalScores | None" = None
```
(`EvalScores` is defined later in the file, so the quoted annotation is required.)

- [ ] **Step 5: Retype `eval_scores` in `src/regintel/state.py`**

Change the import line to add `EvalScores`:
```python
from regintel.types import (
    EvalScores, Finding, Impact, QueryType, Report, RetrievalFilters, RetrievedChunk,
)
```
Change the slot:
```python
    eval_scores: EvalScores | None   # Phase 3
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add src/regintel/types.py src/regintel/state.py tests/test_evaluator.py
git commit -m "feat: add EvalScores type and Report.eval field"
```

---

## Task 2: EvaluatorAgent (`agents/evaluator.py`)

**Files:** Create `src/regintel/agents/evaluator.py`; extend `tests/test_evaluator.py`

- [ ] **Step 1: Append failing tests to `tests/test_evaluator.py`**

```python
from regintel.agents.evaluator import Evaluator, FAITHFULNESS_THRESHOLD
from regintel.types import Citation, Report


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
    assert round(scores.faithfulness, 2) == 0.67   # 2/3 supported
    assert round(scores.citation_coverage, 2) == 0.67  # 2/3 cited
    assert scores.flagged is True  # 0.67 < threshold


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
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: FAIL (ModuleNotFoundError: regintel.agents.evaluator).

- [ ] **Step 3: Implement `src/regintel/agents/evaluator.py`**

```python
from regintel.llm.base import ChatMessage, LLMError
from regintel.types import Citation, EvalScores, Report

FAITHFULNESS_THRESHOLD = 0.7

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "has_citation": {"type": "boolean"},
                },
                "required": ["claim", "supported", "has_citation"],
            },
        }
    },
    "required": ["claims"],
}

_CONFLICTS_SCHEMA = {
    "type": "object",
    "properties": {
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            },
        }
    },
    "required": ["conflicts"],
}

_CLAIMS_SYSTEM = (
    "You are a strict faithfulness judge. Decompose the ANSWER into atomic factual claims. "
    "For each claim decide: supported = is it supported by the numbered CITATIONS (and only "
    "those, not outside knowledge); has_citation = does the claim carry an inline marker like "
    "[0]. Be conservative: if a claim is not clearly supported by the citations, mark it unsupported."
)

_CONFLICTS_SYSTEM = (
    "You detect material contradictions between source passages. Given the numbered passages, "
    "list any pair that materially contradict each other, each as a short description. "
    "If there are none, return an empty list."
)


def _number(citations: list[Citation]) -> str:
    return "\n".join(f"[{i}] ({c.source}) {c.title}: {c.quote}" for i, c in enumerate(citations))


class Evaluator:
    def __init__(self, provider, model: str) -> None:
        self._provider = provider
        self._model = model

    def evaluate(self, query: str, report: Report) -> EvalScores:
        answer = (report.answer or "").strip()
        if not report.citations or answer.lower().startswith("no relevant regulations"):
            return EvalScores(1.0, 1.0, [], flagged=False, notes="no content to evaluate")

        cites = _number(report.citations)

        # Call 1: claim-level faithfulness + citation coverage.
        try:
            out = self._provider.chat_structured(
                [ChatMessage("system", _CLAIMS_SYSTEM),
                 ChatMessage("user", f"ANSWER:\n{report.answer}\n\nCITATIONS:\n{cites}")],
                schema=_CLAIMS_SCHEMA, model=self._model,
            )
            claims = out.get("claims", [])
        except LLMError as exc:
            return EvalScores(0.0, 0.0, [], flagged=True, notes=f"evaluation failed: {exc}")

        if claims:
            supported = sum(1 for c in claims if c.get("supported"))
            cited = sum(1 for c in claims if c.get("has_citation"))
            faithfulness = supported / len(claims)
            citation_coverage = cited / len(claims)
        else:
            faithfulness = 1.0
            citation_coverage = 1.0

        # Call 2: cross-chunk conflicts (non-fatal on failure).
        conflicts: list[str] = []
        conflict_note = ""
        try:
            cout = self._provider.chat_structured(
                [ChatMessage("system", _CONFLICTS_SYSTEM),
                 ChatMessage("user", f"PASSAGES:\n{cites}")],
                schema=_CONFLICTS_SCHEMA, model=self._model,
            )
            conflicts = [c.get("description", "") for c in cout.get("conflicts", []) if c.get("description")]
        except LLMError as exc:
            conflict_note = f" (conflict check skipped: {exc})"

        flagged = faithfulness < FAITHFULNESS_THRESHOLD or bool(conflicts)
        notes = ("ok" if not flagged else "low faithfulness or conflicts detected") + conflict_note
        return EvalScores(
            faithfulness=round(faithfulness, 3),
            citation_coverage=round(citation_coverage, 3),
            conflicts=conflicts,
            flagged=flagged,
            notes=notes,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: PASS (9 tests total in the file).

- [ ] **Step 5: Commit**

```bash
git add src/regintel/agents/evaluator.py tests/test_evaluator.py
git commit -m "feat: add Evaluator (faithfulness, citation coverage, conflict detection)"
```

---

## Task 3: Graph integration (`evaluate` node)

**Files:** Modify `src/regintel/orchestration/nodes.py`, `src/regintel/orchestration/graph.py`; Test `tests/test_graph_eval.py`

- [ ] **Step 1: Write the failing test**

`tests/test_graph_eval.py`:
```python
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
    assert ev.called_with is not None  # evaluator saw the report
    assert isinstance(final["eval_scores"], EvalScores)
    assert final["report"].eval is final["eval_scores"]


def test_flagged_report_gets_warning():
    final = _graph(_FakeEvaluator()).invoke(new_state("q"))
    assert any("low-confidence" in w for w in final["report"].warnings)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_graph_eval.py -v`
Expected: FAIL (build_graph() got unexpected keyword 'evaluator').

- [ ] **Step 3: Add `make_evaluate_node` to `src/regintel/orchestration/nodes.py`**

Add this import at the top (alongside the existing `from regintel.types import ...`):
```python
from regintel.types import EvalScores
```
(Extend the existing import line; it currently imports `QueryType, Report, RetrievalFilters`.)

Append the node factory (after `make_report_node`):
```python
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
```

- [ ] **Step 4: Wire the node in `src/regintel/orchestration/graph.py`**

Update the `make_*` import to include `make_evaluate_node`:
```python
from regintel.orchestration.nodes import (
    make_analyze_node, make_assess_node, make_classify_node, make_evaluate_node,
    make_report_node, make_retrieve_internal_node, make_retrieve_regulations_node,
    route_after_analyze, route_after_regulations,
)
```
Change `build_graph` signature to accept `evaluator`:
```python
def build_graph(*, retriever, orchestrator, analyst, assessor, reporter, evaluator):
```
Add the node (after the `report` node is added):
```python
    g.add_node("evaluate", make_evaluate_node(evaluator))
```
Replace the line `g.add_edge("report", END)` with:
```python
    g.add_edge("report", "evaluate")
    g.add_edge("evaluate", END)
```
In `build_default_graph`, add the Evaluator import and construct it. Add to the imports inside the function:
```python
    from regintel.agents.evaluator import Evaluator
```
Add to the `build_graph(...)` call (as a new keyword arg):
```python
        evaluator=Evaluator(provider, model=resolve_model(Role.EVALUATOR, settings)),
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_graph_eval.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Run the existing routing test (signature change check)**

Run: `uv run pytest tests/test_graph_routing.py -v`
Expected: FAIL — `test_graph_routing.py`'s `_build` calls `build_graph(...)` without `evaluator`. Fix it: in `tests/test_graph_routing.py`, add a fake evaluator and pass it.

Add this class near the other fakes in `tests/test_graph_routing.py`:
```python
class _NoopEvaluator:
    def evaluate(self, query, report):
        from regintel.types import EvalScores
        return EvalScores(1.0, 1.0, [], flagged=False, notes="ok")
```
And in its `build_graph(...)` call add: `evaluator=_NoopEvaluator(),`

Run: `uv run pytest tests/test_graph_routing.py -v` → Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add src/regintel/orchestration/nodes.py src/regintel/orchestration/graph.py tests/test_graph_eval.py tests/test_graph_routing.py
git commit -m "feat: wire Evaluator as final graph node (report -> evaluate -> END)"
```

---

## Task 4: CLI display + e2e/live tests

**Files:** Modify `src/regintel/cli.py`, `tests/test_graph_e2e.py`, `tests/test_ask_live.py`

- [ ] **Step 1: Update the e2e stub + assertion in `tests/test_graph_e2e.py`**

The existing `_StubProvider.chat_structured` keys on schema props. Add handling for the evaluator schemas — insert these two branches BEFORE the final `return` (the report branch):
```python
        if "claims" in props:
            return {"claims": [{"claim": "Gap in blackout windows", "supported": True, "has_citation": True}]}
        if "conflicts" in props:
            return {"conflicts": []}
```
Also pass an evaluator into the `build_graph(...)` call in `test_end_to_end_gap_check_report` by adding the real Evaluator:
```python
    from regintel.agents.evaluator import Evaluator
```
and in the `build_graph(...)` kwargs add: `evaluator=Evaluator(p, "m"),`

Add assertions at the end of `test_end_to_end_gap_check_report`:
```python
    assert report.eval is not None
    assert 0.0 <= report.eval.faithfulness <= 1.0
    assert report.eval.faithfulness == 1.0  # the single stubbed claim is supported
```

- [ ] **Step 2: Run e2e to verify**

Run: `uv run pytest tests/test_graph_e2e.py -v`
Expected: PASS (the graph now runs report → evaluate; eval attached).

- [ ] **Step 3: Add the `Evaluation` section to `cmd_ask` in `src/regintel/cli.py`**

In `cmd_ask`, after the `Impacts` printing block and before the `Warnings` block, add:
```python
    if report.eval is not None:
        e = report.eval
        flag = "  [⚠ FLAGGED]" if e.flagged else ""
        print(f"\nEvaluation: faithfulness={e.faithfulness:.2f}  "
              f"citation_coverage={e.citation_coverage:.2f}  conflicts={len(e.conflicts)}{flag}")
        for c in e.conflicts:
            print(f"  conflict: {c}")
```

- [ ] **Step 4: Strengthen `tests/test_ask_live.py`**

After the existing assertions in `test_ask_live_gap_check`, add:
```python
    assert report.eval is not None
    assert 0.0 <= report.eval.faithfulness <= 1.0
    assert 0.0 <= report.eval.citation_coverage <= 1.0
```

- [ ] **Step 5: Full suite + lint**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all non-live tests PASS; ruff clean. Fix any lint.

- [ ] **Step 6: Commit**

```bash
git add src/regintel/cli.py tests/test_graph_e2e.py tests/test_ask_live.py
git commit -m "feat: show evaluation scores in ask CLI; assert eval in e2e/live tests"
```

- [ ] **Step 7 (controller): live smoke**

Run (requires Ollama): `uv run pytest tests/test_ask_live.py -m live -v`
Expected: PASS — report carries an `EvalScores` with faithfulness in [0,1].

---

## Self-Review (completed by planner)

**Spec coverage:**
- Custom LLM-as-judge, no new deps → Tasks 2 (reuses OllamaProvider) ✅
- EvalScores type + Report.eval + state retype → Task 1 ✅
- Faithfulness + citation coverage (claims call) → Task 2 ✅
- Conflict detection (conflicts call, non-fatal) → Task 2 ✅
- Threshold + flagged logic → Task 2 (`FAITHFULNESS_THRESHOLD = 0.7`) ✅
- Short-circuit (no citations / no-evidence answer) → Task 2 + tests ✅
- Fail-safe on error → Task 2 (claims-call failure) + Task 3 (node-level) ✅
- Graph `evaluate` node, report→evaluate→END → Task 3 ✅
- build_default_graph wires frontier Evaluator → Task 3 ✅
- Flag → warning appended → Task 3 ✅
- CLI Evaluation section → Task 4 ✅
- Tests unit/graph/e2e/live → Tasks 2,3,4 ✅

**Placeholder scan:** none — all code complete.

**Type consistency:** `EvalScores(faithfulness, citation_coverage, conflicts, flagged, notes)` constructed identically in Task 2 (evaluator), Task 3 (node fail-safe + fake), Task 4 (stub). `build_graph(..., evaluator)` keyword matches the routing-test fix (Task 3 Step 6) and e2e (Task 4). `Evaluator(provider, model)` constructor consistent across graph wiring and tests. `Report.eval` attribute used consistently in node, CLI, and tests.

**Note:** Task 3 Step 6 intentionally fixes the pre-existing `test_graph_routing.py` to match the new `build_graph` signature — this is required, not optional.
```
