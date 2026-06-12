# RegIntel Phase 2 Implementation Plan (Orchestration + Reasoning)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the LangGraph multi-agent core — Orchestrator (dynamic query-type routing), Analyst (clause + gap analysis), ImpactAssessor (affected policies + severity), Reporter (cited report) — on top of the Phase 0+1 retrieval spine.

**Architecture:** A `StateGraph(AgentState)` routes by classified query type (LOOKUP / GAP_CHECK / IMPACT) through retrieve → analyze → assess → report nodes with conditional short-circuits. Agents are small classes using `provider.chat_structured`; citations are resolved in code from real retrieved chunks (never model-invented). One Ollama provider, per-role model from `router.py`.

**Tech Stack:** LangGraph, existing `regintel` modules (RetrieverAgent, OllamaProvider, QdrantStore, router), pytest.

**Conventions for the engineer:**
- Run via `uv run` (dev tools in `[dependency-groups]`; plain `uv run pytest`). Do NOT use `--extra dev`.
- Commit per task with the exact message. If you must change `pyproject.toml`/`uv.lock` outside Task 1, STOP and report it.
- **Parallel-build note:** Tasks 3, 4, 5, 6 (the four agents) touch disjoint files and depend only on Task 2's types. They may be implemented in parallel by separate agents that do NOT run git (the controller commits). Tasks 1-2 and 7+ are sequential.

---

## File structure

```
src/regintel/
  types.py                  # MODIFY: + QueryType, Citation, Finding, Impact, Report, cite()
  state.py                  # MODIFY: + query_type, internal; typed slots
  agents/
    orchestrator.py         # CREATE: Orchestrator.classify
    analyst.py              # CREATE: Analyst.analyze
    impact_assessor.py      # CREATE: ImpactAssessor.assess
    reporter.py             # CREATE: Reporter.report
  orchestration/
    __init__.py             # CREATE: empty
    nodes.py                # CREATE: node factories + routing fns
    graph.py                # CREATE: build_graph, build_default_graph, run_query
  cli.py                    # MODIFY: + `ask` command
tests/
  test_phase2_types.py  test_orchestrator.py  test_analyst.py
  test_impact_assessor.py  test_reporter.py
  test_graph_routing.py  test_graph_e2e.py  test_ask_live.py
```

---

## Task 1: Add LangGraph dependency

**Files:** Modify `pyproject.toml`, `uv.lock`

- [ ] **Step 1: Add the dependency**

Run:
```bash
uv add "langgraph>=0.2"
```
Expected: resolves and updates `pyproject.toml` `[project].dependencies` + `uv.lock`.

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from langgraph.graph import StateGraph, START, END; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Verify suite still green**

Run: `uv run pytest -q`
Expected: `45 passed, 2 deselected`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add langgraph dependency"
```

---

## Task 2: Phase 2 data contracts (`types.py`, `state.py`)

**Files:** Modify `src/regintel/types.py`, `src/regintel/state.py`; Test `tests/test_phase2_types.py`

- [ ] **Step 1: Write the failing test**

`tests/test_phase2_types.py`:
```python
from regintel.types import (
    Citation, Finding, Impact, QueryType, Report, RetrievedChunk, cite,
)
from regintel.state import new_state


def test_query_type_values():
    assert QueryType("lookup") is QueryType.LOOKUP
    assert QueryType.GAP_CHECK.value == "gap_check"
    assert QueryType.IMPACT.value == "impact"


def test_cite_from_chunk():
    chunk = RetrievedChunk(
        doc_id="d1", chunk_index=2, text="x" * 400, score=0.5,
        payload={"title": "Policy A", "source": "internal", "url": None},
    )
    c = cite(chunk)
    assert isinstance(c, Citation)
    assert c.doc_id == "d1" and c.chunk_index == 2
    assert c.title == "Policy A" and c.source == "internal"
    assert len(c.quote) <= 300


def test_finding_and_impact_defaults():
    f = Finding(topic="t", requirement="r", internal_status="absent", gap=True, explanation="e")
    assert f.citations == []
    im = Impact(topic="t", affected_policies=["Policy A"], severity="high", rationale="r")
    assert im.severity == "high"


def test_report_defaults():
    r = Report(query_type=QueryType.LOOKUP, answer="hello")
    assert r.citations == [] and r.findings == [] and r.impacts == [] and r.warnings == []


def test_new_state_has_phase2_slots():
    s = new_state("q")
    assert s["internal"] == []
    assert s["retrieved"] == []
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_phase2_types.py -v`
Expected: FAIL (ImportError: cannot import name 'QueryType').

- [ ] **Step 3: Extend `src/regintel/types.py`**

Add at the top (keep existing `RetrievalFilters`, `RetrievedChunk`, `Source`, `DocType`):
```python
from enum import Enum
```
Append to the file:
```python
class QueryType(str, Enum):
    LOOKUP = "lookup"
    GAP_CHECK = "gap_check"
    IMPACT = "impact"


@dataclass
class Citation:
    doc_id: str
    chunk_index: int
    title: str
    source: str
    url: str | None
    quote: str


def cite(chunk: "RetrievedChunk", *, max_quote: int = 300) -> Citation:
    """Build a Citation from a retrieved chunk (quote truncated)."""
    p = chunk.payload or {}
    return Citation(
        doc_id=chunk.doc_id,
        chunk_index=chunk.chunk_index,
        title=p.get("title", ""),
        source=p.get("source", ""),
        url=p.get("url"),
        quote=chunk.text[:max_quote],
    )


@dataclass
class Finding:
    topic: str
    requirement: str
    internal_status: str
    gap: bool
    explanation: str
    citations: list[Citation] = field(default_factory=list)


Severity = Literal["low", "medium", "high", "critical"]


@dataclass
class Impact:
    topic: str
    affected_policies: list[str]
    severity: str
    rationale: str


@dataclass
class Report:
    query_type: QueryType
    answer: str
    citations: list[Citation] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    impacts: list[Impact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
```
(`dataclass`, `field`, `Literal` are already imported in `types.py`; if `Literal` is not, add it to the existing `from typing import ...` line.)

- [ ] **Step 4: Update `src/regintel/state.py`**

Replace the file with:
```python
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
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_phase2_types.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add src/regintel/types.py src/regintel/state.py tests/test_phase2_types.py
git commit -m "feat: add Phase 2 data contracts (QueryType, Citation, Finding, Impact, Report)"
```

---

## Task 3: Orchestrator (`agents/orchestrator.py`)  *(parallel-safe)*

**Files:** Create `src/regintel/agents/orchestrator.py`; Test `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

`tests/test_orchestrator.py`:
```python
from regintel.agents.orchestrator import Orchestrator
from regintel.types import QueryType


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        self.calls.append(messages)
        return self._payload


def test_classify_returns_enum():
    p = _FakeProvider({"query_type": "impact", "reasoning": "a rule changed"})
    assert Orchestrator(p, model="m").classify("How does the new rule affect us?") is QueryType.IMPACT


def test_classify_gap_check():
    p = _FakeProvider({"query_type": "gap_check", "reasoning": "compliance check"})
    assert Orchestrator(p, model="m").classify("Does our policy comply?") is QueryType.GAP_CHECK


def test_classify_bad_value_falls_back_to_gap_check():
    p = _FakeProvider({"query_type": "nonsense", "reasoning": "x"})
    assert Orchestrator(p, model="m").classify("q") is QueryType.GAP_CHECK


def test_classify_missing_key_falls_back():
    p = _FakeProvider({"reasoning": "x"})
    assert Orchestrator(p, model="m").classify("q") is QueryType.GAP_CHECK
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `src/regintel/agents/orchestrator.py`**

```python
from regintel.llm.base import ChatMessage
from regintel.types import QueryType

_SCHEMA = {
    "type": "object",
    "properties": {
        "query_type": {"type": "string", "enum": ["lookup", "gap_check", "impact"]},
        "reasoning": {"type": "string"},
    },
    "required": ["query_type", "reasoning"],
}

_SYSTEM = (
    "You classify a user's regulatory-compliance question into exactly one type:\n"
    "- lookup: a factual question about what a regulation says. "
    "Example: 'What does SEC Rule 10b5-1 require?'\n"
    "- gap_check: asks whether the company's own policies comply with regulation. "
    "Example: 'Does our insider trading policy meet SEC requirements?'\n"
    "- impact: asks how a regulatory change affects the company. "
    "Example: 'A new SEC rule on blackout windows passed — what's the impact on us?'\n"
    "Return the single best type and a one-sentence reasoning."
)


class Orchestrator:
    def __init__(self, provider, model: str) -> None:
        self._provider = provider
        self._model = model

    def classify(self, query: str) -> QueryType:
        out = self._provider.chat_structured(
            [ChatMessage("system", _SYSTEM), ChatMessage("user", query)],
            schema=_SCHEMA,
            model=self._model,
        )
        try:
            return QueryType(out["query_type"])
        except (KeyError, ValueError, TypeError):
            return QueryType.GAP_CHECK
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit** (skip if building in parallel — controller commits)

```bash
git add src/regintel/agents/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add Orchestrator (query-type classification)"
```

---

## Task 4: Analyst (`agents/analyst.py`)  *(parallel-safe)*

**Files:** Create `src/regintel/agents/analyst.py`; Test `tests/test_analyst.py`

- [ ] **Step 1: Write the failing test**

`tests/test_analyst.py`:
```python
from regintel.agents.analyst import Analyst
from regintel.types import RetrievedChunk


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload

    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        return self._payload


def _chunk(doc_id, idx, text, title, source):
    return RetrievedChunk(doc_id=doc_id, chunk_index=idx, text=text, score=0.5,
                          payload={"title": title, "source": source, "url": None})


def test_analyze_resolves_citations_from_indices():
    regs = [_chunk("sec1", 0, "Blackout windows required.", "SEC 8-K", "sec")]
    internal = [_chunk("pol1", 0, "No blackout clause.", "Insider Policy", "internal")]
    payload = {"findings": [{
        "topic": "blackout windows", "requirement": "must define blackout window",
        "internal_status": "absent", "gap": True, "explanation": "policy lacks it",
        "regulation_refs": [0], "internal_refs": [0],
    }]}
    findings = Analyst(_FakeProvider(payload), model="m").analyze("q", regs, internal)
    assert len(findings) == 1
    f = findings[0]
    assert f.gap is True and f.topic == "blackout windows"
    assert {c.source for c in f.citations} == {"sec", "internal"}
    assert any(c.title == "SEC 8-K" for c in f.citations)


def test_analyze_drops_out_of_range_refs():
    regs = [_chunk("sec1", 0, "text", "SEC", "sec")]
    payload = {"findings": [{
        "topic": "t", "requirement": "r", "internal_status": "absent",
        "gap": False, "explanation": "e", "regulation_refs": [5], "internal_refs": [],
    }]}
    findings = Analyst(_FakeProvider(payload), model="m").analyze("q", regs, [])
    assert findings[0].citations == []


def test_analyze_empty_regulations_returns_empty():
    findings = Analyst(_FakeProvider({"findings": []}), model="m").analyze("q", [], [])
    assert findings == []
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_analyst.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `src/regintel/agents/analyst.py`**

```python
from regintel.llm.base import ChatMessage
from regintel.types import Citation, Finding, RetrievedChunk, cite

_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "requirement": {"type": "string"},
                    "internal_status": {"type": "string"},
                    "gap": {"type": "boolean"},
                    "explanation": {"type": "string"},
                    "regulation_refs": {"type": "array", "items": {"type": "integer"}},
                    "internal_refs": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["topic", "requirement", "internal_status", "gap", "explanation"],
            },
        }
    },
    "required": ["findings"],
}

_SYSTEM = (
    "You are a compliance analyst. Compare what the REGULATIONS require against what the "
    "company's INTERNAL DOCUMENTS say. For each relevant topic, state the requirement, the "
    "internal status (or 'absent'), whether there is a gap, and a short explanation. "
    "Cite supporting passages by their integer index using regulation_refs and internal_refs. "
    "Use only the provided passages."
)


def _number(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(none)"
    return "\n".join(f"[{i}] {c.text[:600]}".replace("\n", " ") for i, c in enumerate(chunks))


def _resolve(indices, chunks: list[RetrievedChunk]) -> list[Citation]:
    out: list[Citation] = []
    for idx in indices or []:
        if isinstance(idx, int) and 0 <= idx < len(chunks):
            out.append(cite(chunks[idx]))
    return out


class Analyst:
    def __init__(self, provider, model: str) -> None:
        self._provider = provider
        self._model = model

    def analyze(self, query: str, regulations: list[RetrievedChunk],
                internal: list[RetrievedChunk]) -> list[Finding]:
        if not regulations:
            return []
        user = (
            f"Question: {query}\n\nREGULATIONS:\n{_number(regulations)}\n\n"
            f"INTERNAL DOCUMENTS:\n{_number(internal)}"
        )
        out = self._provider.chat_structured(
            [ChatMessage("system", _SYSTEM), ChatMessage("user", user)],
            schema=_SCHEMA, model=self._model,
        )
        findings: list[Finding] = []
        for f in out.get("findings", []):
            citations = _resolve(f.get("regulation_refs"), regulations) + \
                _resolve(f.get("internal_refs"), internal)
            findings.append(Finding(
                topic=f.get("topic", ""),
                requirement=f.get("requirement", ""),
                internal_status=f.get("internal_status", ""),
                gap=bool(f.get("gap", False)),
                explanation=f.get("explanation", ""),
                citations=citations,
            ))
        return findings
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_analyst.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit** (skip if building in parallel — controller commits)

```bash
git add src/regintel/agents/analyst.py tests/test_analyst.py
git commit -m "feat: add Analyst (clause extraction + gap analysis)"
```

---

## Task 5: ImpactAssessor (`agents/impact_assessor.py`)  *(parallel-safe)*

**Files:** Create `src/regintel/agents/impact_assessor.py`; Test `tests/test_impact_assessor.py`

- [ ] **Step 1: Write the failing test**

`tests/test_impact_assessor.py`:
```python
from regintel.agents.impact_assessor import ImpactAssessor
from regintel.types import Finding, RetrievedChunk


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload

    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        return self._payload


def _internal(title):
    return RetrievedChunk(doc_id=title, chunk_index=0, text="t", score=0.5,
                          payload={"title": title, "source": "internal", "url": None})


def _gap_finding(topic="blackout windows"):
    return Finding(topic=topic, requirement="r", internal_status="absent",
                   gap=True, explanation="e")


def test_assess_returns_impacts_and_validates_policies():
    payload = {"impacts": [{
        "topic": "blackout windows",
        "affected_policies": ["Insider Policy", "Nonexistent Policy"],
        "severity": "high", "rationale": "material gap",
    }]}
    internal = [_internal("Insider Policy")]
    impacts = ImpactAssessor(_FakeProvider(payload), model="m").assess([_gap_finding()], internal)
    assert len(impacts) == 1
    assert impacts[0].affected_policies == ["Insider Policy"]  # unknown dropped
    assert impacts[0].severity == "high"


def test_assess_coerces_bad_severity_to_medium():
    payload = {"impacts": [{"topic": "blackout windows", "affected_policies": [],
                            "severity": "catastrophic", "rationale": "r"}]}
    impacts = ImpactAssessor(_FakeProvider(payload), model="m").assess([_gap_finding()], [])
    assert impacts[0].severity == "medium"


def test_assess_no_gaps_returns_empty():
    non_gap = Finding(topic="t", requirement="r", internal_status="present",
                      gap=False, explanation="e")
    impacts = ImpactAssessor(_FakeProvider({"impacts": []}), model="m").assess([non_gap], [])
    assert impacts == []
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_impact_assessor.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `src/regintel/agents/impact_assessor.py`**

```python
from regintel.llm.base import ChatMessage
from regintel.types import Finding, Impact, RetrievedChunk

_VALID_SEVERITY = {"low", "medium", "high", "critical"}

_SCHEMA = {
    "type": "object",
    "properties": {
        "impacts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "affected_policies": {"type": "array", "items": {"type": "string"}},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "rationale": {"type": "string"},
                },
                "required": ["topic", "affected_policies", "severity", "rationale"],
            },
        }
    },
    "required": ["impacts"],
}

_SYSTEM = (
    "You assess the business impact of compliance gaps. For each gap, name which of the "
    "company's internal policies are affected (choose only from the provided policy titles), "
    "assign a severity (low, medium, high, critical), and give a one-sentence rationale."
)


class ImpactAssessor:
    def __init__(self, provider, model: str) -> None:
        self._provider = provider
        self._model = model

    def assess(self, findings: list[Finding], internal: list[RetrievedChunk]) -> list[Impact]:
        gaps = [f for f in findings if f.gap]
        if not gaps:
            return []
        known = {(c.payload or {}).get("title", "") for c in internal}
        gap_block = "\n".join(
            f"- {f.topic}: requires '{f.requirement}'; internal status: {f.internal_status}"
            for f in gaps
        )
        policy_block = "\n".join(f"- {t}" for t in sorted(known) if t) or "(none)"
        user = f"GAPS:\n{gap_block}\n\nINTERNAL POLICY TITLES:\n{policy_block}"
        out = self._provider.chat_structured(
            [ChatMessage("system", _SYSTEM), ChatMessage("user", user)],
            schema=_SCHEMA, model=self._model,
        )
        impacts: list[Impact] = []
        for im in out.get("impacts", []):
            severity = im.get("severity", "medium")
            if severity not in _VALID_SEVERITY:
                severity = "medium"
            policies = [p for p in im.get("affected_policies", []) if p in known]
            impacts.append(Impact(
                topic=im.get("topic", ""),
                affected_policies=policies,
                severity=severity,
                rationale=im.get("rationale", ""),
            ))
        return impacts
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_impact_assessor.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit** (skip if building in parallel — controller commits)

```bash
git add src/regintel/agents/impact_assessor.py tests/test_impact_assessor.py
git commit -m "feat: add ImpactAssessor (affected policies + severity)"
```

---

## Task 6: Reporter (`agents/reporter.py`)  *(parallel-safe)*

**Files:** Create `src/regintel/agents/reporter.py`; Test `tests/test_reporter.py`

- [ ] **Step 1: Write the failing test**

`tests/test_reporter.py`:
```python
from regintel.agents.reporter import Reporter
from regintel.types import Citation, Finding, QueryType, RetrievedChunk


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload
        self.last_user = None

    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        self.last_user = messages[-1].content
        return self._payload


def _chunk(doc_id, title):
    return RetrievedChunk(doc_id=doc_id, chunk_index=0, text="regulation text", score=0.5,
                          payload={"title": title, "source": "sec", "url": "http://x"})


def test_report_lookup_uses_regulation_chunks():
    provider = _FakeProvider({"answer": "Rule says X [0].", "cited_indices": [0]})
    regs = [_chunk("sec1", "SEC Rule")]
    rep = Reporter(provider, model="m").report("q", QueryType.LOOKUP, [], [], regs, [])
    assert rep.query_type is QueryType.LOOKUP
    assert "Rule says X" in rep.answer
    assert len(rep.citations) == 1 and rep.citations[0].title == "SEC Rule"


def test_report_keeps_only_cited_indices():
    provider = _FakeProvider({"answer": "Only first [0].", "cited_indices": [0]})
    regs = [_chunk("sec1", "First"), _chunk("sec2", "Second")]
    rep = Reporter(provider, model="m").report("q", QueryType.LOOKUP, [], [], regs, [])
    assert [c.title for c in rep.citations] == ["First"]


def test_report_no_evidence_short_circuits_without_llm():
    provider = _FakeProvider({"answer": "should not be used"})
    rep = Reporter(provider, model="m").report("q", QueryType.GAP_CHECK, [], [], [], [])
    assert "no relevant regulations" in rep.answer.lower()
    assert provider.last_user is None  # LLM not called


def test_report_includes_findings_and_impacts():
    provider = _FakeProvider({"answer": "Gap found [0].", "cited_indices": [0]})
    finding = Finding(topic="blackout", requirement="r", internal_status="absent",
                      gap=True, explanation="e",
                      citations=[Citation("d", 0, "Policy", "internal", None, "snippet")])
    rep = Reporter(provider, model="m").report("q", QueryType.GAP_CHECK, [finding], [], [], [])
    assert rep.findings == [finding]
    assert len(rep.citations) == 1
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_reporter.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `src/regintel/agents/reporter.py`**

```python
from regintel.llm.base import ChatMessage
from regintel.types import Citation, Finding, Impact, QueryType, Report, RetrievedChunk, cite

_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited_indices": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "cited_indices"],
}

_SYSTEM = (
    "You are a compliance report writer. Using ONLY the numbered citations provided, write a "
    "clear, well-structured answer to the question. Insert inline markers like [0], [1] where "
    "each claim is supported, and return the list of citation indices you used. Do not invent "
    "facts beyond the citations."
)


def _dedupe(citations: list[Citation]) -> list[Citation]:
    seen: set[tuple[str, int]] = set()
    out: list[Citation] = []
    for c in citations:
        key = (c.doc_id, c.chunk_index)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


class Reporter:
    def __init__(self, provider, model: str) -> None:
        self._provider = provider
        self._model = model

    def _pool(self, findings, regulations) -> list[Citation]:
        pool: list[Citation] = []
        for f in findings:
            pool.extend(f.citations)
        pool.extend(cite(ch) for ch in regulations)
        return _dedupe(pool)

    def report(self, query: str, query_type: QueryType, findings: list[Finding],
               impacts: list[Impact], regulations: list[RetrievedChunk],
               internal: list[RetrievedChunk]) -> Report:
        pool = self._pool(findings, regulations)
        if not pool:
            return Report(
                query_type=query_type,
                answer="No relevant regulations found for this question.",
            )
        numbered = "\n".join(f"[{i}] ({c.source}) {c.title}: {c.quote}" for i, c in enumerate(pool))
        summary = ""
        if findings:
            summary += "\nFINDINGS:\n" + "\n".join(
                f"- {f.topic}: {'GAP' if f.gap else 'ok'} — {f.explanation}" for f in findings
            )
        if impacts:
            summary += "\nIMPACTS:\n" + "\n".join(
                f"- {im.topic}: severity={im.severity}; policies={im.affected_policies}" for im in impacts
            )
        user = f"Question: {query}\nQuery type: {query_type.value}{summary}\n\nCITATIONS:\n{numbered}"
        out = self._provider.chat_structured(
            [ChatMessage("system", _SYSTEM), ChatMessage("user", user)],
            schema=_SCHEMA, model=self._model,
        )
        cited = [pool[i] for i in out.get("cited_indices", [])
                 if isinstance(i, int) and 0 <= i < len(pool)]
        return Report(
            query_type=query_type,
            answer=out.get("answer", ""),
            citations=cited,
            findings=findings,
            impacts=impacts,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_reporter.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit** (skip if building in parallel — controller commits)

```bash
git add src/regintel/agents/reporter.py tests/test_reporter.py
git commit -m "feat: add Reporter (cited report assembly)"
```

---

## Task 7: Graph nodes + wiring (`orchestration/`)

**Files:** Create `src/regintel/orchestration/__init__.py`, `src/regintel/orchestration/nodes.py`, `src/regintel/orchestration/graph.py`; Test `tests/test_graph_routing.py`

- [ ] **Step 1: Write the failing test**

`tests/test_graph_routing.py`:
```python
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
    final = graph.invoke(new_state("q"))
    assert analyst.called is False
    assert assessor.called is False


def test_gap_check_with_findings_runs_assessor():
    finding = Finding(topic="t", requirement="r", internal_status="absent",
                      gap=True, explanation="e")
    graph, analyst, assessor = _build(QueryType.GAP_CHECK, [_chunk()], [finding])
    final = graph.invoke(new_state("q"))
    assert analyst.called is True
    assert assessor.called is True


def test_gap_check_no_findings_skips_assessor():
    graph, analyst, assessor = _build(QueryType.GAP_CHECK, [_chunk()], [])
    final = graph.invoke(new_state("q"))
    assert analyst.called is True
    assert assessor.called is False
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_graph_routing.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `src/regintel/orchestration/__init__.py`** (empty file)

- [ ] **Step 4: Implement `src/regintel/orchestration/nodes.py`**

```python
from regintel.state import AgentState
from regintel.types import QueryType, Report, RetrievalFilters


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


def route_after_regulations(state: AgentState) -> str:
    if not state.get("retrieved"):
        return "report"
    if state.get("query_type") == QueryType.LOOKUP:
        return "report"
    return "retrieve_internal"


def route_after_analyze(state: AgentState) -> str:
    return "assess" if state.get("analyst_findings") else "report"
```

- [ ] **Step 5: Implement `src/regintel/orchestration/graph.py`**

```python
from langgraph.graph import END, START, StateGraph

from regintel.config import Settings, get_settings
from regintel.llm.router import Role, resolve_model
from regintel.orchestration.nodes import (
    make_analyze_node, make_assess_node, make_classify_node, make_report_node,
    make_retrieve_internal_node, make_retrieve_regulations_node,
    route_after_analyze, route_after_regulations,
)
from regintel.state import AgentState, new_state
from regintel.types import Report


def build_graph(*, retriever, orchestrator, analyst, assessor, reporter):
    g = StateGraph(AgentState)
    g.add_node("classify", make_classify_node(orchestrator))
    g.add_node("retrieve_regulations", make_retrieve_regulations_node(retriever))
    g.add_node("retrieve_internal", make_retrieve_internal_node(retriever))
    g.add_node("analyze", make_analyze_node(analyst))
    g.add_node("assess", make_assess_node(assessor))
    g.add_node("report", make_report_node(reporter))

    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve_regulations")
    g.add_conditional_edges("retrieve_regulations", route_after_regulations,
                            {"report": "report", "retrieve_internal": "retrieve_internal"})
    g.add_edge("retrieve_internal", "analyze")
    g.add_conditional_edges("analyze", route_after_analyze,
                            {"assess": "assess", "report": "report"})
    g.add_edge("assess", "report")
    g.add_edge("report", END)
    return g.compile()


def build_default_graph(settings: Settings | None = None, *, retriever=None, provider=None):
    """Wire the real agents using Ollama + the existing retrieval stack."""
    settings = settings or get_settings()
    from regintel.agents.analyst import Analyst
    from regintel.agents.impact_assessor import ImpactAssessor
    from regintel.agents.orchestrator import Orchestrator
    from regintel.agents.reporter import Reporter
    from regintel.llm.ollama_provider import OllamaProvider

    if provider is None:
        provider = OllamaProvider(host=settings.ollama_host,
                                  default_model=settings.ollama_chat_model)
    if retriever is None:
        from qdrant_client import QdrantClient
        from regintel.agents.retriever import RetrieverAgent
        from regintel.embeddings.ollama_embedder import OllamaEmbedder
        from regintel.embeddings.sparse import BM25Encoder
        from regintel.store.qdrant_store import QdrantStore
        client = (QdrantClient(path="./qdrant_storage") if settings.qdrant_embedded
                  else QdrantClient(url=settings.qdrant_url))
        retriever = RetrieverAgent(
            store=QdrantStore(client=client),
            dense=OllamaEmbedder(host=settings.ollama_host, model=settings.ollama_embed_model),
            sparse=BM25Encoder(),
            provider=provider, rerank_model=settings.ollama_chat_model,
        )
    chat = resolve_model(Role.ANALYST, settings)
    frontier = resolve_model(Role.IMPACT_ASSESSOR, settings)
    return build_graph(
        retriever=retriever,
        orchestrator=Orchestrator(provider, model=resolve_model(Role.ORCHESTRATOR, settings)),
        analyst=Analyst(provider, model=chat),
        assessor=ImpactAssessor(provider, model=frontier),
        reporter=Reporter(provider, model=resolve_model(Role.REPORTER, settings)),
    )


def run_query(query: str, *, graph) -> Report:
    final = graph.invoke(new_state(query))
    return final["report"]
```

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/test_graph_routing.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add src/regintel/orchestration tests/test_graph_routing.py
git commit -m "feat: add LangGraph orchestration (nodes, routing, graph builder)"
```

---

## Task 8: CLI `ask` + end-to-end test

**Files:** Modify `src/regintel/cli.py`; Test `tests/test_graph_e2e.py`, `tests/test_ask_live.py`

- [ ] **Step 1: Write the failing e2e test**

`tests/test_graph_e2e.py`:
```python
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
        return {"answer": "Gap in blackout windows [0].", "cited_indices": [0]}


class _Retriever:
    def retrieve(self, query, *, filters=None):
        src = getattr(filters, "source", None)
        title = "Insider Policy" if src == "internal" else "SEC Rule"
        return [RetrievedChunk(doc_id=title, chunk_index=0, text="text", score=0.5,
                               payload={"title": title, "source": src or "sec", "url": None})]


def test_end_to_end_gap_check_report():
    from regintel.agents.analyst import Analyst
    from regintel.agents.impact_assessor import ImpactAssessor
    from regintel.agents.orchestrator import Orchestrator
    from regintel.agents.reporter import Reporter

    p = _StubProvider()
    graph = build_graph(
        retriever=_Retriever(),
        orchestrator=Orchestrator(p, "m"), analyst=Analyst(p, "m"),
        assessor=ImpactAssessor(p, "m"), reporter=Reporter(p, "m"),
    )
    report = run_query("Does our insider policy comply?", graph=graph)
    assert isinstance(report, Report)
    assert report.query_type is QueryType.GAP_CHECK
    assert report.findings and report.findings[0].gap is True
    assert report.impacts and report.impacts[0].severity == "high"
    assert report.citations  # at least one resolved citation
    assert "blackout" in report.answer.lower()
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_graph_e2e.py -v`
Expected: FAIL (the test imports work, but assertion fails only if behavior is wrong; it should PASS once agents+graph from Tasks 3-7 exist). If Tasks 3-7 are complete, this should pass immediately — confirm it does.

- [ ] **Step 3: Add the live test `tests/test_ask_live.py`**

```python
import pytest


@pytest.mark.live
def test_ask_live_gap_check():
    """Requires Ollama (bge-m3 + gpt-oss) + embedded Qdrant. Uses gpt-oss for all roles."""
    from qdrant_client import QdrantClient

    from regintel.agents.retriever import RetrieverAgent
    from regintel.config import Settings
    from regintel.embeddings.ollama_embedder import OllamaEmbedder
    from regintel.embeddings.sparse import BM25Encoder
    from regintel.ingest.internal_docs import load_internal_docs
    from regintel.ingest.pipeline import DocInput, ingest_documents
    from regintel.llm.ollama_provider import OllamaProvider
    from regintel.orchestration.graph import build_default_graph, run_query
    from regintel.store.qdrant_store import QdrantStore
    from pathlib import Path

    s = Settings(_env_file=None)
    store = QdrantStore(client=QdrantClient(":memory:"))
    dense = OllamaEmbedder(host=s.ollama_host, model=s.ollama_embed_model)
    sparse = BM25Encoder()
    docs = [DocInput(doc_id=d.doc_id, title=d.title, text=d.text, source=d.source,
                     jurisdiction=d.jurisdiction, doc_type=d.doc_type)
            for d in load_internal_docs(Path("data/internal"))]
    docs.append(DocInput(doc_id="sec1", title="SEC Insider Trading",
                         text="Issuers must define quarterly blackout windows for insiders.",
                         source="sec", jurisdiction="US-SEC", doc_type="filing"))
    ingest_documents(docs, store=store, dense=dense, sparse=sparse)

    provider = OllamaProvider(host=s.ollama_host, default_model=s.ollama_chat_model)
    retriever = RetrieverAgent(store=store, dense=dense, sparse=sparse,
                               provider=provider, rerank_model=s.ollama_chat_model)
    # Force all roles onto the already-pulled gpt-oss model (no frontier pull needed).
    graph = build_default_graph(s, retriever=retriever, provider=provider)
    report = run_query("Does our insider trading policy comply with SEC blackout rules?",
                       graph=graph)
    assert report.answer
    assert report.query_type is not None
```

- [ ] **Step 4: Add the `ask` command to `src/regintel/cli.py`**

Add this import near the others:
```python
from regintel.orchestration.graph import build_default_graph, run_query
```
Add the command function (place after `cmd_query`):
```python
def cmd_ask(args) -> None:
    settings = get_settings()
    graph = build_default_graph(settings)
    report = run_query(args.query, graph=graph)
    print(f"\n=== {report.query_type.value.upper()} ===")
    print(report.answer)
    if report.citations:
        print("\nCitations:")
        for i, c in enumerate(report.citations):
            loc = c.url or f"{c.doc_id}#{c.chunk_index}"
            print(f"  [{i}] ({c.source}) {c.title} — {loc}")
    if report.findings:
        print("\nFindings:")
        for f in report.findings:
            print(f"  - {f.topic}: {'GAP' if f.gap else 'ok'} — {f.explanation}")
    if report.impacts:
        print("\nImpacts:")
        for im in report.impacts:
            print(f"  - {im.topic}: severity={im.severity}; policies={im.affected_policies}")
    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  ! {w}")
```
Register it in `main()` after the `query` subparser block:
```python
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("query")
    p_ask.set_defaults(func=cmd_ask)
```

- [ ] **Step 5: Run the full non-live suite + lint**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: all non-live tests PASS (Phase 1's 45 + Phase 2's new); ruff clean. Fix any lint (e.g. unused imports).

- [ ] **Step 6: Commit**

```bash
git add src/regintel/cli.py tests/test_graph_e2e.py tests/test_ask_live.py
git commit -m "feat: add `ask` CLI command and end-to-end orchestration tests"
```

- [ ] **Step 7 (optional, controller): live smoke**

Run (requires Ollama + ingested corpus): `uv run pytest tests/test_ask_live.py -m live -v`
Expected: PASS. If a frontier model isn't pulled, the assess node degrades gracefully (impacts empty + warning) — the report still returns.

---

## Self-Review (completed by planner)

**Spec coverage:**
- Dynamic routing (LOOKUP/GAP_CHECK/IMPACT) → Task 3 (classify) + Task 7 (routing fns) ✅
- Conditional short-circuits (empty regs, no gaps) → Task 7 `route_after_*` + tests ✅
- Code-resolved citations → Task 4 `_resolve`, Task 6 pool, Task 2 `cite()` ✅
- Typed contracts (QueryType/Citation/Finding/Impact/Report) → Task 2 ✅
- State refinements (query_type, internal, typed slots) → Task 2 ✅
- Analyst gap analysis → Task 4 ✅
- ImpactAssessor severity + policy validation + frontier model → Task 5 + Task 7 model wiring ✅
- Reporter citation pool + only-cited + no-evidence → Task 6 ✅
- Graph module (nodes, graph, run_query, build_default_graph) → Task 7 ✅
- Per-node error handling + warnings surfaced → Task 7 nodes + report node ✅
- CLI `ask` → Task 8 ✅
- Tests (unit/routing/e2e/live) → Tasks 3-8 ✅
- langgraph dep → Task 1 ✅

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `provider.chat_structured(messages, *, schema, model, temperature)` matches the Phase 0 `LLMProvider` protocol. `RetrievedChunk(doc_id, chunk_index, text, score, payload)` matches Phase 0 `types.py`. `RetrievalFilters(jurisdiction=…, source=…)` matches Phase 0. Agent constructors `(provider, model)` are consistent across Tasks 3-6 and the wiring in Task 7. `cite()`/`Citation` fields consistent between Tasks 2, 4, 6. `build_graph(*, retriever, orchestrator, analyst, assessor, reporter)` signature matches the routing test (Task 7) and e2e test (Task 8).

**Parallelization note:** Tasks 3-6 create disjoint files (`agents/orchestrator.py`, `analyst.py`, `impact_assessor.py`, `reporter.py` + their tests) and import only from Task 2's `types.py` + Phase 0 `llm/base.py`. They are safe to build concurrently; the controller commits each after review. Task 7 depends on all of 3-6; Task 8 depends on 7.
```
