# Regulatory Intelligence System — Phase 3 Design (Evaluator / Trust Layer)

**Date:** 2026-06-12
**Status:** Approved (brainstorming) — continuous execution authorized
**Scope:** Phase 3 — the EvaluatorAgent that scores the Reporter's output for faithfulness, citation coverage, and cross-chunk conflicts, and flags low-confidence answers. Builds on the Phase 2 orchestration graph.

---

## 1. Context

Phases 0–2 deliver a working multi-agent pipeline: classify → retrieve → analyze → assess → **report** (grounded, cited). Phase 3 adds the **trust layer**: an Evaluator that grades the report *after* generation and attaches a confidence signal, so a low-faithfulness or self-contradictory answer is flagged rather than presented as authoritative. This is the README's promised "refuses to pretend it's sure when it isn't."

---

## 2. Decisions (locked in brainstorming)

- **Custom LLM-as-judge**, not RAGAS. RAGAS would add ~0.4–0.7 GB of dependencies (`pyarrow`, `pandas`, `scipy`, `datasets`, full `langchain` stack), defaults to OpenAI, and is brittle on our Ollama-cloud stack. The custom judge adds **zero new dependencies** (reuses the hardened `OllamaProvider`). README frames the metrics as "RAGAS-style." RAGAS may be added later as an *optional* extra (`uv add --optional eval ragas`) — out of scope here.
- **Flag + annotate** gating: always return the answer, attach scores, set `flagged=True` + a warning when below threshold. No blocking, no regeneration (YAGNI for now).
- **Frontier model** for the Evaluator (`Role.EVALUATOR` → `kimi-k2.6:cloud`), since judging faithfulness is reasoning-heavy.
- **Fail-safe**: if evaluation itself errors, the answer is `flagged=True` (we never claim confidence we couldn't verify).

---

## 3. Data contract

Added to `src/regintel/types.py`:
```python
@dataclass
class EvalScores:
    faithfulness: float        # 0..1 — fraction of answer claims supported by cited passages
    citation_coverage: float   # 0..1 — fraction of answer claims carrying a citation
    conflicts: list[str]       # cross-chunk contradictions (empty = none)
    flagged: bool              # below threshold OR conflicts present
    notes: str                 # brief judge rationale
```
`Report` gains a field: `eval: EvalScores | None = None`.
`AgentState.eval_scores` is retyped from `dict[str, Any] | None` to `EvalScores | None`.

---

## 4. EvaluatorAgent (`agents/evaluator.py`)

`Evaluator(provider, model)` with `evaluate(query: str, report: Report) -> EvalScores`.

**Short-circuit:** if `report.answer` is the no-evidence sentinel or `report.citations` is empty, return `EvalScores(1.0, 1.0, [], flagged=False, notes="no content to evaluate")` without LLM calls.

**Call 1 — claims** (`chat_structured`): decompose `report.answer` into atomic claims; for each, judge against the numbered cited passages:
```
{"claims": [{"claim": str, "supported": bool, "has_citation": bool}]}
```
- `faithfulness = supported_count / total_claims` (1.0 if no claims).
- `citation_coverage = has_citation_count / total_claims` (1.0 if no claims).

**Call 2 — conflicts** (`chat_structured`): over the numbered cited passages, identify cross-chunk contradictions:
```
{"conflicts": [{"description": str}]}
```
- `conflicts = [c["description"] for c in ...]`.

**Threshold:** module constant `FAITHFULNESS_THRESHOLD = 0.7`. `flagged = faithfulness < FAITHFULNESS_THRESHOLD or bool(conflicts)`.

**Fail-safe:** wrap each call; if a call raises `LLMError` or returns unusable data, degrade that metric conservatively (faithfulness defaults to 0.0 on hard failure) and set `flagged=True`, `notes` explaining the failure.

---

## 5. Graph integration

Insert an `evaluate` node between `report` and `END`:
```
classify → retrieve_regulations → [retrieve_internal → analyze → (assess)] → report → evaluate → END
```
- `orchestration/nodes.py`: `make_evaluate_node(evaluator)` — reads `state["report"]`; computes scores; sets `report.eval = scores`; if `scores.flagged`, appends `"low-confidence: <notes>"` to `report.warnings`; returns `{"eval_scores": scores, "report": report}`. Wrapped in try/except (failure → fail-safe `EvalScores`, never crashes).
- `orchestration/graph.py`: `build_graph(..., evaluator)` adds the node, replaces `add_edge("report", END)` with `add_edge("report", "evaluate")` + `add_edge("evaluate", END)`. `build_default_graph` constructs `Evaluator(provider, model=resolve_model(Role.EVALUATOR, settings))`.

---

## 6. CLI

`regintel ask` prints an **Evaluation** section after the report body:
```
Evaluation: faithfulness=0.86  citation_coverage=0.90  conflicts=0  [⚠ FLAGGED] (if flagged)
```
Conflicts, if any, listed below.

---

## 7. Error handling

Reuses the established per-node try/except pattern. Evaluate-node failure produces fail-safe `EvalScores(faithfulness=0.0, citation_coverage=0.0, conflicts=[], flagged=True, notes="evaluation failed: ...")` and a warning — the report still returns. Structured-output parsing already hardened in `OllamaProvider`.

---

## 8. Testing (TDD)

- **Unit** (`tests/test_evaluator.py`, fake provider): faithfulness math (e.g. 2/3 supported → ≈0.67), citation coverage, conflict parsing, threshold flagging (0.6 → flagged, 0.9 → not), conflicts-present → flagged, short-circuit on empty citations (no LLM call), fail-safe on provider error.
- **Graph** (`tests/test_graph_eval.py` or extend routing): `evaluate` runs after `report`; final `state["eval_scores"]` set and `report.eval` populated; flagged report has a warning.
- **e2e:** extend the stub provider to also return claims/conflicts payloads; assert `report.eval` populated with a float faithfulness.
- **Live** (extend `test_ask_live.py`): assert `report.eval is not None` and `0.0 <= report.eval.faithfulness <= 1.0`.

---

## 9. File structure

```
src/regintel/
  types.py                  # MODIFY: + EvalScores; Report.eval field
  state.py                  # MODIFY: eval_scores: EvalScores | None
  agents/evaluator.py       # CREATE: Evaluator
  orchestration/nodes.py    # MODIFY: + make_evaluate_node
  orchestration/graph.py    # MODIFY: wire evaluate node + evaluator param
  cli.py                    # MODIFY: print Evaluation section
tests/
  test_evaluator.py         # CREATE
  test_graph_eval.py        # CREATE (or extend test_graph_routing.py)
  test_graph_e2e.py         # MODIFY: stub eval payloads, assert report.eval
  test_ask_live.py          # MODIFY: assert report.eval
```

---

## 10. Definition of done

1. `uv run pytest` green (new + existing); ruff clean.
2. `build_graph` includes the `evaluate` node; graph compiles and runs report → evaluate → END.
3. `regintel ask` prints faithfulness, citation coverage, conflicts, and a flag when low-confidence.
4. Live run attaches a real `EvalScores` to the report; a deliberately unsupported answer would be flagged (fail-safe verified).

## 11. Out of scope

RAGAS integration (future optional extra), answer regeneration / blocking gates, ground-truth-based metrics (context recall, answer correctness), MonitorAgent (Phase 4), FastAPI + UI (Phase 5).
