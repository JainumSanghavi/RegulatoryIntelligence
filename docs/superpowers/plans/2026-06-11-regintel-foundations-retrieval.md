# RegIntel Phase 0+1 Implementation Plan (Foundations + Retrieval Spine)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a low-disk, Ollama-served ingestion + hybrid-retrieval spine: live SEC EDGAR docs + synthetic internal docs → Qdrant (dense bge-m3 + sparse BM25) → RRF fusion → LLM reranker → `RetrieverAgent`, behind a pluggable LLM provider layer and typed agent state.

**Architecture:** Python package `regintel` managed by `uv`. A `LLMProvider` protocol abstracts Ollama (default) and Claude. Embeddings: bge-m3 dense via Ollama + FastEmbed BM25 sparse. A single shared Qdrant `corpus` collection (named dense+sparse vectors, rich payload) is queried with the Qdrant Query API (server-side RRF fusion + payload filters), then reranked listwise by an Ollama chat model. TDD throughout; embedded Qdrant `:memory:` for tests so no Docker needed in CI.

**Tech Stack:** Python 3.12, uv, pydantic-settings, httpx, qdrant-client, fastembed, langchain-text-splitters, tiktoken, selectolax, anthropic, pytest, pytest-asyncio, respx (HTTP mocking).

**Conventions for the executing engineer:**
- Run everything through uv: `uv run pytest ...`, `uv run python ...`.
- `uv` lives at `~/.local/bin/uv` (or on PATH). Commit after every task with the shown message.
- Commit author is already configured per-commit in this repo via `-c user.name=... -c user.email=...`; a normal `git commit` works once `git config user.name/email` is set (Task 0 does this).
- Models: `OLLAMA_CHAT_MODEL` default `gpt-oss:120b-cloud` (the pulled cloud alias), `OLLAMA_EMBED_MODEL` default `bge-m3`. Frontier roles default `kimi-k2.6` / `deepseek-v4-pro` but are NOT required for Phase 0+1 tests (those are exercised in Phase 2/3).

---

## File Structure (decomposition)

```
pyproject.toml                      # uv project + deps
docker-compose.yml                  # Qdrant service
.env.example                        # config template
README.md                           # setup + demo
src/regintel/
  __init__.py
  config.py                         # pydantic-settings Settings
  types.py                          # shared dataclasses/TypedDicts (RetrievalFilters, RetrievedChunk, ...)
  state.py                          # AgentState TypedDict (full LangGraph schema)
  llm/
    __init__.py
    base.py                         # LLMProvider protocol, ChatMessage, LLMError
    ollama_provider.py              # OllamaProvider (chat, chat_structured)
    claude_provider.py              # ClaudeProvider
    router.py                       # role -> (provider, model) resolution
  embeddings/
    __init__.py
    ollama_embedder.py              # dense via bge-m3
    sparse.py                       # FastEmbed BM25 sparse encoder
  store/
    __init__.py
    schema.py                       # ChunkPayload, collection constants, vector config
    qdrant_store.py                 # QdrantStore: ensure_collection, upsert, hybrid_search
  ingest/
    __init__.py
    chunker.py                      # chunk_text()
    sec_edgar.py                    # SECClient: live fetch + on-disk cache
    internal_docs.py               # load_internal_docs()
    pipeline.py                     # ingest_documents()
  rerank/
    __init__.py
    llm_reranker.py                 # rerank()
  agents/
    __init__.py
    retriever.py                    # RetrieverAgent
  cli.py                            # `python -m regintel.cli ingest|query`
data/
  internal/                         # synthetic docs (committed)
tests/
  conftest.py
  test_*.py
```

---

## Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/regintel/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`, `.env.example`, `docker-compose.yml`

- [ ] **Step 1: Set git identity (so plain `git commit` works)**

Run:
```bash
git config user.name "Jainum Sanghavi" && git config user.email "sanghavi.j@northeastern.edu"
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "regintel"
version = "0.1.0"
description = "Regulatory Intelligence System - multi-agent RAG over SEC filings + internal docs"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "httpx>=0.27",
    "qdrant-client>=1.12",
    "fastembed>=0.4",
    "langchain-text-splitters>=0.3",
    "tiktoken>=0.7",
    "selectolax>=0.3.21",
    "anthropic>=0.39",
    "tenacity>=8.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
    "ruff>=0.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/regintel"]

[tool.pytest.ini_options]
pythonpath = ["src"]
markers = ["live: tests that require a running Ollama/Qdrant (deselected by default)"]
addopts = "-m 'not live'"
asyncio_mode = "auto"
```

- [ ] **Step 3: Create package + test files**

`src/regintel/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: (empty file)

`tests/test_smoke.py`:
```python
import regintel


def test_version():
    assert regintel.__version__ == "0.1.0"
```

- [ ] **Step 4: Create `.env.example`**

```bash
# Ollama (product runtime)
OLLAMA_HOST=http://localhost:11434
OLLAMA_CHAT_MODEL=gpt-oss:120b-cloud
OLLAMA_EMBED_MODEL=bge-m3
OLLAMA_FRONTIER_MODEL=kimi-k2.6

# Anthropic (optional; only if a role is routed to Claude)
ANTHROPIC_API_KEY=

# SEC EDGAR (required for ingestion) - SEC mandates a descriptive UA
SEC_USER_AGENT=Jainum Sanghavi sanghavi.j@northeastern.edu

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_EMBEDDED=false
```

- [ ] **Step 5: Create `docker-compose.yml`**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

- [ ] **Step 6: Install and run the smoke test**

Run:
```bash
uv sync --extra dev && uv run pytest -v
```
Expected: `test_smoke.py::test_version PASSED`. (uv creates `.venv` and resolves deps.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/ tests/ .env.example docker-compose.yml
git commit -m "chore: scaffold regintel uv project with smoke test"
```

---

## Task 1: Config (`config.py`)

**Files:**
- Create: `src/regintel/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from regintel.config import Settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    s = Settings(_env_file=None)
    assert s.ollama_host == "http://localhost:11434"
    assert s.ollama_chat_model == "gpt-oss:120b-cloud"
    assert s.ollama_embed_model == "bge-m3"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.qdrant_embedded is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "llama3.1")
    monkeypatch.setenv("QDRANT_EMBEDDED", "true")
    s = Settings(_env_file=None)
    assert s.ollama_chat_model == "llama3.1"
    assert s.qdrant_embedded is True
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: regintel.config`).

- [ ] **Step 3: Implement `src/regintel/config.py`**

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    ollama_chat_model: str = "gpt-oss:120b-cloud"
    ollama_embed_model: str = "bge-m3"
    ollama_frontier_model: str = "kimi-k2.6"

    anthropic_api_key: str | None = None

    sec_user_agent: str = "RegIntel Example example@example.com"

    qdrant_url: str = "http://localhost:6333"
    qdrant_embedded: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/regintel/config.py tests/test_config.py
git commit -m "feat: add pydantic-settings config"
```

---

## Task 2: Shared types (`types.py`)

**Files:**
- Create: `src/regintel/types.py`, `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

`tests/test_types.py`:
```python
from regintel.types import RetrievalFilters, RetrievedChunk


def test_filters_to_dict_drops_none():
    f = RetrievalFilters(jurisdiction="US-SEC", source=None)
    assert f.as_payload_conditions() == {"jurisdiction": "US-SEC"}


def test_retrieved_chunk_roundtrip():
    c = RetrievedChunk(
        doc_id="d1", chunk_index=0, text="hello",
        score=0.9, payload={"source": "sec"},
    )
    assert c.doc_id == "d1"
    assert c.score == 0.9
    assert c.payload["source"] == "sec"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/regintel/types.py`**

```python
from dataclasses import dataclass, field
from typing import Any, Literal

Source = Literal["sec", "internal"]
DocType = Literal["filing", "policy", "sop", "contract"]


@dataclass
class RetrievalFilters:
    """Payload filters for retrieval. None fields are ignored."""
    jurisdiction: str | None = None
    doc_type: str | None = None
    source: str | None = None
    date_from: str | None = None  # ISO date, filters filed_date >=
    date_to: str | None = None    # ISO date, filters filed_date <=

    def as_payload_conditions(self) -> dict[str, str]:
        """Equality conditions only (date range handled separately by the store)."""
        out: dict[str, str] = {}
        for key in ("jurisdiction", "doc_type", "source"):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        return out


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_index: int
    text: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    rerank_rationale: str | None = None
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_types.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/types.py tests/test_types.py
git commit -m "feat: add shared retrieval types"
```

---

## Task 3: Agent state (`state.py`)

**Files:**
- Create: `src/regintel/state.py`, `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

`tests/test_state.py`:
```python
from regintel.state import AgentState, new_state


def test_new_state_minimal():
    s = new_state("Does our insider trading policy comply with SEC rules?")
    assert s["query"].startswith("Does our")
    assert s["retrieved"] == []
    assert s["errors"] == []


def test_agentstate_is_dict():
    s: AgentState = new_state("q")
    s["sub_questions"] = ["a", "b"]
    assert s["sub_questions"] == ["a", "b"]
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/state.py`**

```python
from typing import Any, TypedDict

from regintel.types import RetrievalFilters, RetrievedChunk


class AgentState(TypedDict, total=False):
    """Full LangGraph state. Phases 0+1 populate only query/filters/retrieved/errors."""
    query: str
    sub_questions: list[str]
    filters: RetrievalFilters
    retrieved: list[RetrievedChunk]
    analyst_findings: list[dict[str, Any]]      # Phase 2
    impact_assessments: list[dict[str, Any]]    # Phase 2
    report: dict[str, Any] | None               # Phase 2
    eval_scores: dict[str, Any] | None          # Phase 3
    errors: list[str]
    messages: list[dict[str, Any]]


def new_state(query: str) -> AgentState:
    return AgentState(
        query=query,
        sub_questions=[],
        filters=RetrievalFilters(),
        retrieved=[],
        errors=[],
        messages=[],
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/state.py tests/test_state.py
git commit -m "feat: add typed AgentState"
```

---

## Task 4: LLM provider base (`llm/base.py`)

**Files:**
- Create: `src/regintel/llm/__init__.py` (empty), `src/regintel/llm/base.py`, `tests/test_llm_base.py`

- [ ] **Step 1: Write the failing test**

`tests/test_llm_base.py`:
```python
from regintel.llm.base import ChatMessage, LLMError, LLMProvider


def test_chat_message():
    m = ChatMessage(role="user", content="hi")
    assert m.as_dict() == {"role": "user", "content": "hi"}


def test_provider_is_protocol():
    # A minimal stub satisfies the protocol structurally.
    class Stub:
        def chat(self, messages, *, model=None, temperature=0.0, **kw): return "ok"
        def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw): return {}

    p: LLMProvider = Stub()
    assert p.chat([]) == "ok"


def test_llm_error():
    assert issubclass(LLMError, Exception)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_llm_base.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement files**

`src/regintel/llm/__init__.py`: (empty)

`src/regintel/llm/base.py`:
```python
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class LLMError(Exception):
    """Raised when an LLM provider call fails terminally."""


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@runtime_checkable
class LLMProvider(Protocol):
    def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str: ...

    def chat_structured(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


def _normalize(messages: list[ChatMessage] | list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for m in messages:
        out.append(m.as_dict() if isinstance(m, ChatMessage) else dict(m))
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_llm_base.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/llm/__init__.py src/regintel/llm/base.py tests/test_llm_base.py
git commit -m "feat: add LLMProvider protocol and base types"
```

---

## Task 5: Ollama provider (`llm/ollama_provider.py`)

**Files:**
- Create: `src/regintel/llm/ollama_provider.py`, `tests/test_ollama_provider.py`

- [ ] **Step 1: Write the failing test (HTTP mocked with respx)**

`tests/test_ollama_provider.py`:
```python
import httpx
import respx

from regintel.llm.base import ChatMessage
from regintel.llm.ollama_provider import OllamaProvider


@respx.mock
def test_chat_returns_content():
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "hello world"}})
    )
    p = OllamaProvider(host="http://localhost:11434", default_model="gpt-oss:120b-cloud")
    out = p.chat([ChatMessage("user", "hi")])
    assert out == "hello world"
    assert route.called
    sent = route.calls.last.request
    assert b'"stream": false' in sent.content or b'"stream":false' in sent.content


@respx.mock
def test_chat_structured_parses_json_content():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": '{"score": 5}'}})
    )
    p = OllamaProvider(host="http://localhost:11434", default_model="m")
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
    out = p.chat_structured([ChatMessage("user", "rate")], schema=schema)
    assert out == {"score": 5}


@respx.mock
def test_chat_raises_llmerror_on_500():
    from regintel.llm.base import LLMError
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(500))
    p = OllamaProvider(host="http://localhost:11434", default_model="m", max_retries=1)
    import pytest
    with pytest.raises(LLMError):
        p.chat([ChatMessage("user", "hi")])
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_ollama_provider.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/llm/ollama_provider.py`**

```python
import json
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from regintel.llm.base import ChatMessage, LLMError, _normalize


class OllamaProvider:
    """LLMProvider backed by Ollama's native /api/chat endpoint."""

    def __init__(
        self,
        host: str,
        default_model: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._host = host.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        self._max_retries = max_retries

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        @retry(
            retry=retry_if_exception_type((httpx.HTTPError,)),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.5, max=8),
            reraise=True,
        )
        def _do() -> dict[str, Any]:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._host}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()

        try:
            return _do()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama chat failed: {exc}") from exc

    def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        payload = {
            "model": model or self._default_model,
            "messages": _normalize(messages),
            "stream": False,
            "options": {"temperature": temperature},
        }
        data = self._post_chat(payload)
        return data["message"]["content"]

    def chat_structured(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = {
            "model": model or self._default_model,
            "messages": _normalize(messages),
            "stream": False,
            "format": schema,  # Ollama structured outputs: JSON schema
            "options": {"temperature": temperature},
        }
        data = self._post_chat(payload)
        content = data["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Ollama returned non-JSON structured content: {content!r}") from exc
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ollama_provider.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/regintel/llm/ollama_provider.py tests/test_ollama_provider.py
git commit -m "feat: add OllamaProvider with retries and structured output"
```

---

## Task 6: Claude provider (`llm/claude_provider.py`)

**Files:**
- Create: `src/regintel/llm/claude_provider.py`, `tests/test_claude_provider.py`

- [ ] **Step 1: Write the failing test (SDK mocked)**

`tests/test_claude_provider.py`:
```python
from regintel.llm.base import ChatMessage
from regintel.llm.claude_provider import ClaudeProvider


class _FakeMessages:
    def create(self, **kw):
        class _Block:
            text = '{"ok": true}' if kw.get("_structured") else "plain text"
        class _Resp:
            content = [_Block()]
        # mimic anthropic: system separated, messages list present
        assert "messages" in kw
        return _Resp()


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_chat_extracts_text():
    p = ClaudeProvider(api_key="x", default_model="claude-sonnet-4-6", client=_FakeClient())
    assert p.chat([ChatMessage("user", "hi")]) == "plain text"


def test_chat_separates_system():
    p = ClaudeProvider(api_key="x", default_model="m", client=_FakeClient())
    out = p.chat([ChatMessage("system", "be brief"), ChatMessage("user", "hi")])
    assert out == "plain text"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_claude_provider.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/llm/claude_provider.py`**

```python
import json
from typing import Any

from regintel.llm.base import ChatMessage, LLMError, _normalize


class ClaudeProvider:
    """LLMProvider backed by the Anthropic SDK. Optional in Phase 0+1."""

    def __init__(
        self,
        api_key: str | None,
        default_model: str,
        *,
        max_tokens: int = 4096,
        client: Any = None,
    ) -> None:
        self._default_model = default_model
        self._max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise LLMError("anthropic SDK not installed") from exc
            self._client = anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def _split(messages):
        msgs = _normalize(messages)
        system = "\n".join(m["content"] for m in msgs if m["role"] == "system") or None
        convo = [m for m in msgs if m["role"] != "system"]
        return system, convo

    def chat(self, messages, *, model=None, temperature=0.0, **kwargs) -> str:
        system, convo = self._split(messages)
        kw = {
            "model": model or self._default_model,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system:
            kw["system"] = system
        try:
            resp = self._client.messages.create(**kw)
            return resp.content[0].text
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Claude chat failed: {exc}") from exc

    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kwargs) -> dict:
        # Append a schema instruction; Claude returns JSON text we parse.
        msgs = list(messages)
        instruction = ChatMessage(
            "user",
            f"Respond ONLY with JSON matching this schema, no prose:\n{json.dumps(schema)}",
        )
        msgs.append(instruction)
        text = self.chat(msgs, model=model, temperature=temperature, _structured=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Claude returned non-JSON: {text!r}") from exc
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_claude_provider.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/llm/claude_provider.py tests/test_claude_provider.py
git commit -m "feat: add ClaudeProvider adapter"
```

---

## Task 7: Provider router (`llm/router.py`)

**Files:**
- Create: `src/regintel/llm/router.py`, `tests/test_router.py`

- [ ] **Step 1: Write the failing test**

`tests/test_router.py`:
```python
from regintel.config import Settings
from regintel.llm.router import Role, resolve_model, ROLE_TIER


def test_role_tiers():
    assert ROLE_TIER[Role.ANALYST] == "chat"
    assert ROLE_TIER[Role.IMPACT_ASSESSOR] == "frontier"
    assert ROLE_TIER[Role.EVALUATOR] == "frontier"
    assert ROLE_TIER[Role.RERANKER] == "chat"


def test_resolve_model_uses_settings():
    s = Settings(_env_file=None)
    assert resolve_model(Role.ANALYST, s) == s.ollama_chat_model
    assert resolve_model(Role.IMPACT_ASSESSOR, s) == s.ollama_frontier_model
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_router.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/llm/router.py`**

```python
from enum import Enum

from regintel.config import Settings
from regintel.llm.base import LLMProvider
from regintel.llm.ollama_provider import OllamaProvider


class Role(str, Enum):
    ORCHESTRATOR = "orchestrator"
    ANALYST = "analyst"
    IMPACT_ASSESSOR = "impact_assessor"
    REPORTER = "reporter"
    EVALUATOR = "evaluator"
    RERANKER = "reranker"


# "chat" -> gpt-oss; "frontier" -> kimi/deepseek
ROLE_TIER: dict[Role, str] = {
    Role.ORCHESTRATOR: "chat",
    Role.ANALYST: "chat",
    Role.REPORTER: "chat",
    Role.RERANKER: "chat",
    Role.IMPACT_ASSESSOR: "frontier",
    Role.EVALUATOR: "frontier",
}


def resolve_model(role: Role, settings: Settings) -> str:
    tier = ROLE_TIER[role]
    return settings.ollama_frontier_model if tier == "frontier" else settings.ollama_chat_model


def get_provider(settings: Settings) -> LLMProvider:
    """Phase 0+1: always Ollama. Claude swap is a future per-role extension."""
    return OllamaProvider(host=settings.ollama_host, default_model=settings.ollama_chat_model)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_router.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/llm/router.py tests/test_router.py
git commit -m "feat: add role-based model router"
```

---

## Task 8: Store schema (`store/schema.py`)

**Files:**
- Create: `src/regintel/store/__init__.py` (empty), `src/regintel/store/schema.py`, `tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

`tests/test_schema.py`:
```python
from regintel.store.schema import (
    COLLECTION, DENSE_DIM, DENSE_VEC, SPARSE_VEC, ChunkPayload, point_id,
)


def test_constants():
    assert COLLECTION == "corpus"
    assert DENSE_DIM == 1024
    assert DENSE_VEC == "dense"
    assert SPARSE_VEC == "sparse"


def test_point_id_deterministic():
    a = point_id("docA", 3)
    b = point_id("docA", 3)
    c = point_id("docA", 4)
    assert a == b
    assert a != c


def test_chunk_payload_to_dict():
    p = ChunkPayload(
        doc_id="d1", chunk_index=0, text="t", source="sec",
        jurisdiction="US-SEC", doc_type="filing", title="10-K",
        form_type="10-K", accession_no="000", filed_date="2026-01-01",
    )
    d = p.as_dict()
    assert d["source"] == "sec"
    assert d["form_type"] == "10-K"
    assert d["url"] is None
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_schema.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/store/schema.py`**

```python
import uuid
from dataclasses import asdict, dataclass

COLLECTION = "corpus"
CHANGELOG_COLLECTION = "regulation_changelog"  # reserved for Phase 4
DENSE_VEC = "dense"
SPARSE_VEC = "sparse"
DENSE_DIM = 1024  # bge-m3

_NAMESPACE = uuid.UUID("11111111-1111-1111-1111-111111111111")


def point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic UUID5 so re-ingest updates rather than duplicates."""
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{chunk_index}"))


@dataclass
class ChunkPayload:
    doc_id: str
    chunk_index: int
    text: str
    source: str           # "sec" | "internal"
    jurisdiction: str     # "US-SEC" | "internal"
    doc_type: str         # "filing" | "policy" | "sop" | "contract"
    title: str
    url: str | None = None
    regulation_id: str | None = None
    form_type: str | None = None
    accession_no: str | None = None
    effective_date: str | None = None
    filed_date: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)
```

`src/regintel/store/__init__.py`: (empty)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/store/__init__.py src/regintel/store/schema.py tests/test_schema.py
git commit -m "feat: add Qdrant payload schema and deterministic point ids"
```

---

## Task 9: Qdrant store (`store/qdrant_store.py`)

**Files:**
- Create: `src/regintel/store/qdrant_store.py`, `tests/test_qdrant_store.py`

> Tests use the in-process client `QdrantClient(":memory:")` — no Docker needed. Sparse vectors are supplied directly as (indices, values) so this test does not depend on the embedder.

- [ ] **Step 1: Write the failing test**

`tests/test_qdrant_store.py`:
```python
from qdrant_client import QdrantClient

from regintel.store.qdrant_store import QdrantStore, ChunkRecord
from regintel.store.schema import ChunkPayload


def _record(doc_id, idx, dense, sparse_idx, sparse_val, **payload_kw):
    payload = ChunkPayload(
        doc_id=doc_id, chunk_index=idx, text=f"text {idx}",
        source=payload_kw.get("source", "sec"),
        jurisdiction=payload_kw.get("jurisdiction", "US-SEC"),
        doc_type=payload_kw.get("doc_type", "filing"),
        title="T",
    )
    return ChunkRecord(
        payload=payload, dense=dense,
        sparse_indices=sparse_idx, sparse_values=sparse_val,
    )


def test_ensure_collection_and_upsert_count():
    store = QdrantStore(client=QdrantClient(":memory:"))
    store.ensure_collection()
    recs = [
        _record("d1", 0, [0.1] * 1024, [1, 5], [0.7, 0.3]),
        _record("d1", 1, [0.2] * 1024, [2, 5], [0.6, 0.4]),
    ]
    store.upsert(recs)
    assert store.count() == 2
    # Idempotent: re-upsert same records does not duplicate.
    store.upsert(recs)
    assert store.count() == 2


def test_hybrid_search_filters_by_jurisdiction():
    store = QdrantStore(client=QdrantClient(":memory:"))
    store.ensure_collection()
    store.upsert([
        _record("sec1", 0, [0.9] + [0.0] * 1023, [1], [1.0], jurisdiction="US-SEC"),
        _record("int1", 0, [0.9] + [0.0] * 1023, [1], [1.0],
                jurisdiction="internal", source="internal", doc_type="policy"),
    ])
    from regintel.types import RetrievalFilters
    results = store.hybrid_search(
        dense=[0.9] + [0.0] * 1023,
        sparse_indices=[1], sparse_values=[1.0],
        filters=RetrievalFilters(jurisdiction="US-SEC"),
        limit=10,
    )
    assert len(results) == 1
    assert results[0].payload["jurisdiction"] == "US-SEC"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_qdrant_store.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/store/qdrant_store.py`**

```python
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from regintel.store.schema import (
    COLLECTION, DENSE_DIM, DENSE_VEC, SPARSE_VEC, ChunkPayload, point_id,
)
from regintel.types import RetrievalFilters, RetrievedChunk


@dataclass
class ChunkRecord:
    payload: ChunkPayload
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str = COLLECTION) -> None:
        self._client = client
        self._collection = collection

    @classmethod
    def from_settings(cls, settings) -> "QdrantStore":
        if settings.qdrant_embedded:
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(url=settings.qdrant_url)
        return cls(client)

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={DENSE_VEC: qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE)},
            sparse_vectors_config={SPARSE_VEC: qm.SparseVectorParams()},
        )

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count

    def upsert(self, records: list[ChunkRecord]) -> None:
        points = []
        for r in records:
            points.append(
                qm.PointStruct(
                    id=point_id(r.payload.doc_id, r.payload.chunk_index),
                    vector={
                        DENSE_VEC: r.dense,
                        SPARSE_VEC: qm.SparseVector(indices=r.sparse_indices, values=r.sparse_values),
                    },
                    payload=r.payload.as_dict(),
                )
            )
        self._client.upsert(self._collection, points=points)

    def _build_filter(self, filters: RetrievalFilters) -> qm.Filter | None:
        must: list[qm.FieldCondition] = []
        for key, val in filters.as_payload_conditions().items():
            must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=val)))
        if filters.date_from or filters.date_to:
            rng = qm.DatetimeRange(gte=filters.date_from, lte=filters.date_to)
            must.append(qm.FieldCondition(key="filed_date", range=rng))
        return qm.Filter(must=must) if must else None

    def hybrid_search(
        self,
        *,
        dense: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        filters: RetrievalFilters | None = None,
        limit: int = 20,
        prefetch_limit: int = 50,
    ) -> list[RetrievedChunk]:
        qfilter = self._build_filter(filters) if filters else None
        prefetch = [
            qm.Prefetch(query=dense, using=DENSE_VEC, limit=prefetch_limit, filter=qfilter),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse_indices, values=sparse_values),
                using=SPARSE_VEC, limit=prefetch_limit, filter=qfilter,
            ),
        ]
        resp = self._client.query_points(
            self._collection,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        out: list[RetrievedChunk] = []
        for p in resp.points:
            payload = p.payload or {}
            out.append(
                RetrievedChunk(
                    doc_id=payload.get("doc_id", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    text=payload.get("text", ""),
                    score=p.score,
                    payload=payload,
                )
            )
        return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_qdrant_store.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/regintel/store/qdrant_store.py tests/test_qdrant_store.py
git commit -m "feat: add QdrantStore with hybrid RRF search and filters"
```

---

## Task 10: Dense embedder (`embeddings/ollama_embedder.py`)

**Files:**
- Create: `src/regintel/embeddings/__init__.py` (empty), `src/regintel/embeddings/ollama_embedder.py`, `tests/test_ollama_embedder.py`

- [ ] **Step 1: Write the failing test (HTTP mocked)**

`tests/test_ollama_embedder.py`:
```python
import httpx
import respx

from regintel.embeddings.ollama_embedder import OllamaEmbedder


@respx.mock
def test_embed_batch_returns_vectors():
    respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    )
    emb = OllamaEmbedder(host="http://localhost:11434", model="bge-m3")
    out = emb.embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]


@respx.mock
def test_embed_one():
    respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.5, 0.6]]})
    )
    emb = OllamaEmbedder(host="http://localhost:11434", model="bge-m3")
    assert emb.embed_one("hello") == [0.5, 0.6]
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_ollama_embedder.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/embeddings/ollama_embedder.py`**

```python
import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from regintel.llm.base import LLMError


class OllamaEmbedder:
    """Dense embeddings via Ollama /api/embed (bge-m3)."""

    def __init__(self, host: str, model: str, *, timeout: float = 60.0, max_retries: int = 3) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

    def embed(self, texts: list[str]) -> list[list[float]]:
        @retry(
            retry=retry_if_exception_type((httpx.HTTPError,)),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.5, max=8),
            reraise=True,
        )
        def _do() -> list[list[float]]:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._host}/api/embed",
                    json={"model": self._model, "input": texts},
                )
                resp.raise_for_status()
                return resp.json()["embeddings"]

        try:
            return _do()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama embed failed: {exc}") from exc

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
```

`src/regintel/embeddings/__init__.py`: (empty)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ollama_embedder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/embeddings/__init__.py src/regintel/embeddings/ollama_embedder.py tests/test_ollama_embedder.py
git commit -m "feat: add Ollama dense embedder (bge-m3)"
```

---

## Task 11: Sparse BM25 encoder (`embeddings/sparse.py`)

**Files:**
- Create: `src/regintel/embeddings/sparse.py`, `tests/test_sparse.py`

> FastEmbed's `Bm25` is non-neural (tokenizer + IDF); first use downloads a small (~KB-MB) artifact. The test is marked `live` because it touches FastEmbed's model cache; a pure-unit test covers the adapter shape with a fake.

- [ ] **Step 1: Write the failing test**

`tests/test_sparse.py`:
```python
import pytest

from regintel.embeddings.sparse import BM25Encoder, SparseVec


class _FakeEmbedding:
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values


class _FakeModel:
    def embed(self, texts):
        for i, _ in enumerate(texts):
            yield _FakeEmbedding([i, i + 1], [0.5, 0.5])

    def query_embed(self, text):
        yield _FakeEmbedding([0], [1.0])


def test_encode_documents_with_injected_model():
    enc = BM25Encoder(model=_FakeModel())
    out = enc.encode(["doc one", "doc two"])
    assert isinstance(out[0], SparseVec)
    assert out[0].indices == [0, 1]
    assert out[1].indices == [1, 2]


def test_encode_query_with_injected_model():
    enc = BM25Encoder(model=_FakeModel())
    q = enc.encode_query("find this")
    assert q.indices == [0]
    assert q.values == [1.0]


@pytest.mark.live
def test_real_bm25_loads():
    enc = BM25Encoder()  # downloads Qdrant/bm25
    out = enc.encode(["insider trading blackout window policy"])
    assert len(out[0].indices) > 0
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_sparse.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/embeddings/sparse.py`**

```python
from dataclasses import dataclass


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class BM25Encoder:
    """Sparse lexical vectors via FastEmbed BM25 (non-neural)."""

    def __init__(self, model=None, model_name: str = "Qdrant/bm25") -> None:
        if model is not None:
            self._model = model
        else:
            from fastembed import SparseTextEmbedding
            self._model = SparseTextEmbedding(model_name=model_name)

    def encode(self, texts: list[str]) -> list[SparseVec]:
        out: list[SparseVec] = []
        for emb in self._model.embed(texts):
            out.append(SparseVec(indices=list(emb.indices), values=list(emb.values)))
        return out

    def encode_query(self, text: str) -> SparseVec:
        # query_embed applies query-side weighting; fall back to embed if absent.
        if hasattr(self._model, "query_embed"):
            emb = next(iter(self._model.query_embed(text)))
        else:
            emb = next(iter(self._model.embed([text])))
        return SparseVec(indices=list(emb.indices), values=list(emb.values))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_sparse.py -v`
Expected: PASS (2 non-live tests; the `live` one is deselected).

- [ ] **Step 5: Commit**

```bash
git add src/regintel/embeddings/sparse.py tests/test_sparse.py
git commit -m "feat: add FastEmbed BM25 sparse encoder"
```

---

## Task 12: Chunker (`ingest/chunker.py`)

**Files:**
- Create: `src/regintel/ingest/__init__.py` (empty), `src/regintel/ingest/chunker.py`, `tests/test_chunker.py`

- [ ] **Step 1: Write the failing test**

`tests/test_chunker.py`:
```python
from regintel.ingest.chunker import Chunk, chunk_text


def test_chunk_text_indices_sequential():
    text = "word " * 2000
    chunks = chunk_text(text, doc_id="d1", chunk_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_short_text_single_chunk():
    chunks = chunk_text("just a short bit", doc_id="d2")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/ingest/chunker.py`**

```python
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str


def _token_len(text: str) -> int:
    return len(_ENC.encode(text))


def chunk_text(
    text: str,
    *,
    doc_id: str,
    chunk_tokens: int = 800,
    overlap_tokens: int = 150,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = [p for p in splitter.split_text(text) if p.strip()]
    if not pieces:
        return []
    return [Chunk(doc_id=doc_id, chunk_index=i, text=p) for i, p in enumerate(pieces)]
```

`src/regintel/ingest/__init__.py`: (empty)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/ingest/__init__.py src/regintel/ingest/chunker.py tests/test_chunker.py
git commit -m "feat: add token-aware text chunker"
```

---

## Task 13: SEC EDGAR client (`ingest/sec_edgar.py`)

**Files:**
- Create: `src/regintel/ingest/sec_edgar.py`, `tests/test_sec_edgar.py`, `tests/fixtures/sec_filing.html`

- [ ] **Step 1: Create the HTML fixture**

`tests/fixtures/sec_filing.html`:
```html
<html><head><title>10-K</title></head>
<body>
<p>Item 1A. Risk Factors</p>
<p>Our insider trading policy prohibits trading on material non-public information.</p>
<script>ignore me</script>
</body></html>
```

- [ ] **Step 2: Write the failing test**

`tests/test_sec_edgar.py`:
```python
from pathlib import Path

import httpx
import respx

from regintel.ingest.sec_edgar import SECClient, SECFiling


def test_html_to_text_strips_markup():
    html = Path("tests/fixtures/sec_filing.html").read_text()
    text = SECClient.html_to_text(html)
    assert "Risk Factors" in text
    assert "insider trading policy" in text
    assert "ignore me" not in text
    assert "<p>" not in text


@respx.mock
def test_fetch_document_uses_user_agent_and_caches(tmp_path):
    url = "https://www.sec.gov/Archives/edgar/data/1/x.htm"
    route = respx.get(url).mock(
        return_value=httpx.Response(200, text="<html><body><p>Hello SEC</p></body></html>")
    )
    client = SECClient(user_agent="Tester test@example.com", cache_dir=tmp_path)
    text1 = client.fetch_document(url)
    assert "Hello SEC" in text1
    assert route.calls.last.request.headers["user-agent"] == "Tester test@example.com"
    # Second call served from cache (no new HTTP call).
    text2 = client.fetch_document(url)
    assert text2 == text1
    assert route.call_count == 1


@respx.mock
def test_full_text_search_parses_hits():
    respx.get(url__startswith="https://efts.sec.gov/LATEST/search-index").mock(
        return_value=httpx.Response(200, json={
            "hits": {"hits": [
                {"_id": "0001-24-000001:doc.htm",
                 "_source": {"display_names": ["ACME (CIK 0000001)"],
                             "form": "8-K", "file_date": "2026-05-01"}}
            ]}
        })
    )
    client = SECClient(user_agent="Tester test@example.com")
    hits = client.full_text_search("insider trading", forms=["8-K"], limit=1)
    assert len(hits) == 1
    assert isinstance(hits[0], SECFiling)
    assert hits[0].form_type == "8-K"
    assert hits[0].filed_date == "2026-05-01"
```

- [ ] **Step 3: Run to verify fail**

Run: `uv run pytest tests/test_sec_edgar.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement `src/regintel/ingest/sec_edgar.py`**

```python
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from selectolax.parser import HTMLParser
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_FTS_URL = "https://efts.sec.gov/LATEST/search-index"


@dataclass
class SECFiling:
    accession_no: str
    title: str
    form_type: str
    filed_date: str
    doc_url: str | None = None


class SECClient:
    """Live SEC EDGAR access with on-disk cache and rate limiting (~10 req/s)."""

    def __init__(self, user_agent: str, cache_dir: Path | None = None, *, min_interval: float = 0.12) -> None:
        self._headers = {"User-Agent": user_agent}
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = min_interval
        self._last_call = 0.0

    @staticmethod
    def html_to_text(html: str) -> str:
        tree = HTMLParser(html)
        for tag in tree.css("script, style"):
            tag.decompose()
        body = tree.body or tree
        text = body.text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def _cache_path(self, url: str) -> Path | None:
        if not self._cache_dir:
            return None
        return self._cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".txt")

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def _get(self, url: str, **kwargs):
        self._throttle()
        with httpx.Client(timeout=30.0, headers=self._headers, follow_redirects=True) as client:
            resp = client.get(url, **kwargs)
            resp.raise_for_status()
            return resp

    def fetch_document(self, url: str) -> str:
        cache = self._cache_path(url)
        if cache and cache.exists():
            return cache.read_text()
        text = self.html_to_text(self._get(url).text)
        if cache:
            cache.write_text(text)
        return text

    def full_text_search(self, query: str, *, forms: list[str] | None = None, limit: int = 10) -> list[SECFiling]:
        params = {"q": query, "from": 0}
        if forms:
            params["forms"] = ",".join(forms)
        data = self._get(_FTS_URL, params=params).json()
        hits = data.get("hits", {}).get("hits", [])[:limit]
        out: list[SECFiling] = []
        for h in hits:
            src = h.get("_source", {})
            names = src.get("display_names") or ["Unknown"]
            out.append(
                SECFiling(
                    accession_no=h.get("_id", "").split(":")[0],
                    title=names[0],
                    form_type=src.get("form", ""),
                    filed_date=src.get("file_date", ""),
                )
            )
        return out
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_sec_edgar.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/regintel/ingest/sec_edgar.py tests/test_sec_edgar.py tests/fixtures/sec_filing.html
git commit -m "feat: add SEC EDGAR client with cache, throttle, full-text search"
```

---

## Task 14: Synthetic internal docs + loader (`ingest/internal_docs.py`)

**Files:**
- Create: `data/internal/insider_trading_policy.md`, `data/internal/data_retention_sop.md`, `data/internal/vendor_contract.md`, `src/regintel/ingest/internal_docs.py`, `tests/test_internal_docs.py`

- [ ] **Step 1: Author the synthetic corpus (deliberate gaps vs SEC)**

`data/internal/insider_trading_policy.md`:
```markdown
# ACME Corp Insider Trading Policy
Employees may not trade ACME securities while aware of material non-public information (MNPI).
Violations may result in termination. This policy applies to all employees and directors.

## Pre-clearance
Officers must obtain pre-clearance from the General Counsel before trading.

<!-- DELIBERATE GAP: no defined quarterly blackout window around earnings; no 10b5-1 plan guidance -->
```

`data/internal/data_retention_sop.md`:
```markdown
# ACME Records Retention SOP
Financial records are retained for 5 years. Email is retained for 1 year.

<!-- DELIBERATE GAP: SEC Rule 17a-4 implies certain broker-dealer records require longer, WORM storage -->
```

`data/internal/vendor_contract.md`:
```markdown
# Master Services Agreement (excerpt)
Vendor shall maintain confidentiality of ACME data. Either party may terminate with 30 days notice.
Vendor processes ACME customer data in the course of services.

<!-- DELIBERATE OVERLAP: data-handling clauses relevant to disclosure/controls obligations -->
```

- [ ] **Step 2: Write the failing test**

`tests/test_internal_docs.py`:
```python
from pathlib import Path

from regintel.ingest.internal_docs import load_internal_docs


def test_load_internal_docs_reads_corpus():
    docs = load_internal_docs(Path("data/internal"))
    titles = {d.doc_id for d in docs}
    assert "insider_trading_policy" in titles
    assert len(docs) == 3
    pol = next(d for d in docs if d.doc_id == "insider_trading_policy")
    assert pol.doc_type == "policy"
    assert pol.source == "internal"
    assert "MNPI" in pol.text


def test_doc_type_inferred_from_filename():
    docs = load_internal_docs(Path("data/internal"))
    types = {d.doc_id: d.doc_type for d in docs}
    assert types["data_retention_sop"] == "sop"
    assert types["vendor_contract"] == "contract"
```

- [ ] **Step 3: Run to verify fail**

Run: `uv run pytest tests/test_internal_docs.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement `src/regintel/ingest/internal_docs.py`**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InternalDoc:
    doc_id: str
    title: str
    text: str
    doc_type: str
    source: str = "internal"
    jurisdiction: str = "internal"


def _infer_doc_type(stem: str) -> str:
    if "sop" in stem:
        return "sop"
    if "contract" in stem or "agreement" in stem:
        return "contract"
    return "policy"


def load_internal_docs(directory: Path) -> list[InternalDoc]:
    docs: list[InternalDoc] = []
    for path in sorted(Path(directory).glob("*.md")):
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        docs.append(
            InternalDoc(
                doc_id=path.stem,
                title=title,
                text=text,
                doc_type=_infer_doc_type(path.stem),
            )
        )
    return docs
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_internal_docs.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add data/internal src/regintel/ingest/internal_docs.py tests/test_internal_docs.py
git commit -m "feat: add synthetic internal corpus and loader"
```

---

## Task 15: Ingestion pipeline (`ingest/pipeline.py`)

**Files:**
- Create: `src/regintel/ingest/pipeline.py`, `tests/test_pipeline.py`

> The pipeline glues chunker + embedders + store. Tests inject fakes so no network/model is needed.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
from regintel.ingest.pipeline import DocInput, ingest_documents
from regintel.embeddings.sparse import SparseVec


class _FakeDense:
    def embed(self, texts):
        return [[0.1] * 1024 for _ in texts]


class _FakeSparse:
    def encode(self, texts):
        return [SparseVec([1], [1.0]) for _ in texts]


class _FakeStore:
    def __init__(self):
        self.records = []
    def ensure_collection(self):
        self.ensured = True
    def upsert(self, records):
        self.records.extend(records)


def test_ingest_chunks_embeds_upserts():
    store = _FakeStore()
    docs = [
        DocInput(doc_id="d1", title="T", text="word " * 1000, source="sec",
                 jurisdiction="US-SEC", doc_type="filing", form_type="10-K"),
    ]
    n = ingest_documents(docs, store=store, dense=_FakeDense(), sparse=_FakeSparse(),
                         chunk_tokens=100, overlap_tokens=20)
    assert store.ensured is True
    assert n == len(store.records) > 1
    rec = store.records[0]
    assert rec.payload.source == "sec"
    assert rec.payload.form_type == "10-K"
    assert len(rec.dense) == 1024
    assert rec.sparse_indices == [1]
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/ingest/pipeline.py`**

```python
from dataclasses import dataclass

from regintel.ingest.chunker import chunk_text
from regintel.store.qdrant_store import ChunkRecord
from regintel.store.schema import ChunkPayload


@dataclass
class DocInput:
    doc_id: str
    title: str
    text: str
    source: str
    jurisdiction: str
    doc_type: str
    url: str | None = None
    regulation_id: str | None = None
    form_type: str | None = None
    accession_no: str | None = None
    effective_date: str | None = None
    filed_date: str | None = None


def ingest_documents(
    docs: list[DocInput],
    *,
    store,
    dense,
    sparse,
    chunk_tokens: int = 800,
    overlap_tokens: int = 150,
) -> int:
    store.ensure_collection()
    records: list[ChunkRecord] = []
    for doc in docs:
        chunks = chunk_text(
            doc.text, doc_id=doc.doc_id,
            chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
        )
        if not chunks:
            continue
        texts = [c.text for c in chunks]
        dense_vecs = dense.embed(texts)
        sparse_vecs = sparse.encode(texts)
        for chunk, dvec, svec in zip(chunks, dense_vecs, sparse_vecs, strict=True):
            payload = ChunkPayload(
                doc_id=doc.doc_id, chunk_index=chunk.chunk_index, text=chunk.text,
                source=doc.source, jurisdiction=doc.jurisdiction, doc_type=doc.doc_type,
                title=doc.title, url=doc.url, regulation_id=doc.regulation_id,
                form_type=doc.form_type, accession_no=doc.accession_no,
                effective_date=doc.effective_date, filed_date=doc.filed_date,
            )
            records.append(
                ChunkRecord(payload=payload, dense=dvec,
                            sparse_indices=svec.indices, sparse_values=svec.values)
            )
    if records:
        store.upsert(records)
    return len(records)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/ingest/pipeline.py tests/test_pipeline.py
git commit -m "feat: add ingestion pipeline (chunk -> embed -> upsert)"
```

---

## Task 16: LLM reranker (`rerank/llm_reranker.py`)

**Files:**
- Create: `src/regintel/rerank/__init__.py` (empty), `src/regintel/rerank/llm_reranker.py`, `tests/test_reranker.py`

- [ ] **Step 1: Write the failing test**

`tests/test_reranker.py`:
```python
from regintel.rerank.llm_reranker import rerank
from regintel.types import RetrievedChunk


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload
    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        return self._payload


def _chunks():
    return [
        RetrievedChunk("d1", 0, "irrelevant text", 0.5, {"doc_id": "d1"}),
        RetrievedChunk("d2", 0, "highly relevant text", 0.4, {"doc_id": "d2"}),
    ]


def test_rerank_reorders_and_limits():
    provider = _FakeProvider({"ranking": [
        {"index": 1, "rationale": "directly answers"},
        {"index": 0, "rationale": "tangential"},
    ]})
    out = rerank("q", _chunks(), provider=provider, model="m", top_k=1)
    assert len(out) == 1
    assert out[0].doc_id == "d2"
    assert out[0].rerank_rationale == "directly answers"


def test_rerank_ignores_out_of_range_indices():
    provider = _FakeProvider({"ranking": [{"index": 99, "rationale": "x"}, {"index": 0, "rationale": "y"}]})
    out = rerank("q", _chunks(), provider=provider, model="m", top_k=5)
    assert [c.doc_id for c in out] == ["d1"]
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_reranker.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/rerank/llm_reranker.py`**

```python
from regintel.llm.base import ChatMessage
from regintel.types import RetrievedChunk

_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["index", "rationale"],
            },
        }
    },
    "required": ["ranking"],
}

_SYSTEM = (
    "You are a search reranker. Given a query and numbered candidate passages, "
    "return them ordered most-relevant first. Use only the provided passages."
)


def _format_candidates(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks):
        snippet = c.text[:600].replace("\n", " ")
        lines.append(f"[{i}] {snippet}")
    return "\n".join(lines)


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    provider,
    model: str,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    user = (
        f"Query: {query}\n\nCandidates:\n{_format_candidates(chunks)}\n\n"
        f"Return a JSON ranking (best first) of the candidate indices with a short rationale each."
    )
    result = provider.chat_structured(
        [ChatMessage("system", _SYSTEM), ChatMessage("user", user)],
        schema=_RERANK_SCHEMA, model=model,
    )
    ordered: list[RetrievedChunk] = []
    seen: set[int] = set()
    for item in result.get("ranking", []):
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(chunks) or idx in seen:
            continue
        seen.add(idx)
        chunk = chunks[idx]
        chunk.rerank_rationale = item.get("rationale")
        ordered.append(chunk)
    # Append any chunks the model omitted, preserving original order.
    for i, c in enumerate(chunks):
        if i not in seen:
            ordered.append(c)
    return ordered[:top_k]
```

`src/regintel/rerank/__init__.py`: (empty)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_reranker.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/rerank/__init__.py src/regintel/rerank/llm_reranker.py tests/test_reranker.py
git commit -m "feat: add LLM listwise reranker"
```

---

## Task 17: RetrieverAgent (`agents/retriever.py`)

**Files:**
- Create: `src/regintel/agents/__init__.py` (empty), `src/regintel/agents/retriever.py`, `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test**

`tests/test_retriever.py`:
```python
from regintel.agents.retriever import RetrieverAgent
from regintel.embeddings.sparse import SparseVec
from regintel.types import RetrievalFilters, RetrievedChunk


class _FakeDense:
    def embed_one(self, text):
        return [0.1] * 1024


class _FakeSparse:
    def encode_query(self, text):
        return SparseVec([1], [1.0])


class _FakeStore:
    def __init__(self, results):
        self._results = results
        self.last_filters = None
    def hybrid_search(self, *, dense, sparse_indices, sparse_values, filters, limit, **kw):
        self.last_filters = filters
        return self._results


class _FakeProvider:
    def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw):
        return {"ranking": [{"index": 0, "rationale": "best"}]}


def test_retrieve_runs_full_pipeline():
    candidates = [
        RetrievedChunk("d1", 0, "relevant", 0.5, {"doc_id": "d1"}),
        RetrievedChunk("d2", 0, "less", 0.4, {"doc_id": "d2"}),
    ]
    agent = RetrieverAgent(
        store=_FakeStore(candidates), dense=_FakeDense(), sparse=_FakeSparse(),
        provider=_FakeProvider(), rerank_model="m", top_k=1,
    )
    out = agent.retrieve("insider trading", filters=RetrievalFilters(jurisdiction="US-SEC"))
    assert len(out) == 1
    assert out[0].doc_id == "d1"
    assert out[0].rerank_rationale == "best"


def test_retrieve_passes_filters_to_store():
    store = _FakeStore([])
    agent = RetrieverAgent(store=store, dense=_FakeDense(), sparse=_FakeSparse(),
                           provider=_FakeProvider(), rerank_model="m")
    agent.retrieve("q", filters=RetrievalFilters(source="internal"))
    assert store.last_filters.source == "internal"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/regintel/agents/retriever.py`**

```python
from regintel.rerank.llm_reranker import rerank
from regintel.types import RetrievalFilters, RetrievedChunk


class RetrieverAgent:
    """Hybrid retrieve (dense+sparse RRF) then LLM rerank."""

    def __init__(
        self, *, store, dense, sparse, provider, rerank_model: str,
        candidate_limit: int = 20, top_k: int = 8,
    ) -> None:
        self._store = store
        self._dense = dense
        self._sparse = sparse
        self._provider = provider
        self._rerank_model = rerank_model
        self._candidate_limit = candidate_limit
        self._top_k = top_k

    def retrieve(self, query: str, *, filters: RetrievalFilters | None = None) -> list[RetrievedChunk]:
        filters = filters or RetrievalFilters()
        dense_vec = self._dense.embed_one(query)
        sparse_vec = self._sparse.encode_query(query)
        candidates = self._store.hybrid_search(
            dense=dense_vec,
            sparse_indices=sparse_vec.indices,
            sparse_values=sparse_vec.values,
            filters=filters,
            limit=self._candidate_limit,
        )
        if not candidates:
            return []
        return rerank(
            query, candidates,
            provider=self._provider, model=self._rerank_model, top_k=self._top_k,
        )
```

`src/regintel/agents/__init__.py`: (empty)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regintel/agents/__init__.py src/regintel/agents/retriever.py tests/test_retriever.py
git commit -m "feat: add RetrieverAgent (hybrid + rerank)"
```

---

## Task 18: CLI wiring (`cli.py`) + full-stack live smoke test

**Files:**
- Create: `src/regintel/cli.py`, `tests/test_cli_live.py`

> The CLI is the real wiring (real Ollama, real Qdrant). Its test is `live` (deselected by default). This is the task that proves the whole spine works end-to-end.

- [ ] **Step 1: Write the live smoke test**

`tests/test_cli_live.py`:
```python
import pytest

from regintel.config import Settings


@pytest.mark.live
def test_end_to_end_ingest_and_query():
    """Requires Ollama (bge-m3 + gpt-oss) running and Qdrant embedded."""
    from qdrant_client import QdrantClient
    from regintel.store.qdrant_store import QdrantStore
    from regintel.embeddings.ollama_embedder import OllamaEmbedder
    from regintel.embeddings.sparse import BM25Encoder
    from regintel.ingest.pipeline import DocInput, ingest_documents
    from regintel.agents.retriever import RetrieverAgent
    from regintel.llm.ollama_provider import OllamaProvider
    from regintel.types import RetrievalFilters

    s = Settings(_env_file=None)
    store = QdrantStore(client=QdrantClient(":memory:"))
    dense = OllamaEmbedder(host=s.ollama_host, model=s.ollama_embed_model)
    sparse = BM25Encoder()
    docs = [
        DocInput(doc_id="d1", title="Insider Trading", source="sec", jurisdiction="US-SEC",
                 doc_type="filing", text="Insider trading on material non-public information is prohibited."),
        DocInput(doc_id="d2", title="Office Supplies", source="sec", jurisdiction="US-SEC",
                 doc_type="filing", text="We purchase staplers and paper for the office."),
    ]
    n = ingest_documents(docs, store=store, dense=dense, sparse=sparse)
    assert n >= 2
    provider = OllamaProvider(host=s.ollama_host, default_model=s.ollama_chat_model)
    agent = RetrieverAgent(store=store, dense=dense, sparse=sparse,
                           provider=provider, rerank_model=s.ollama_chat_model, top_k=1)
    out = agent.retrieve("rules about trading on insider information",
                         filters=RetrievalFilters(jurisdiction="US-SEC"))
    assert out
    assert out[0].doc_id == "d1"
```

- [ ] **Step 2: Implement `src/regintel/cli.py`**

```python
import argparse

from qdrant_client import QdrantClient

from regintel.agents.retriever import RetrieverAgent
from regintel.config import get_settings
from regintel.embeddings.ollama_embedder import OllamaEmbedder
from regintel.embeddings.sparse import BM25Encoder
from regintel.ingest.internal_docs import load_internal_docs
from regintel.ingest.pipeline import DocInput, ingest_documents
from regintel.ingest.sec_edgar import SECClient
from regintel.llm.ollama_provider import OllamaProvider
from regintel.store.qdrant_store import QdrantStore
from regintel.types import RetrievalFilters
from pathlib import Path


def _build(settings):
    if settings.qdrant_embedded:
        client = QdrantClient(path="./qdrant_storage")
    else:
        client = QdrantClient(url=settings.qdrant_url)
    store = QdrantStore(client=client)
    dense = OllamaEmbedder(host=settings.ollama_host, model=settings.ollama_embed_model)
    sparse = BM25Encoder()
    return store, dense, sparse


def cmd_ingest(args) -> None:
    settings = get_settings()
    store, dense, sparse = _build(settings)
    docs: list[DocInput] = []
    # Internal docs
    for d in load_internal_docs(Path("data/internal")):
        docs.append(DocInput(doc_id=d.doc_id, title=d.title, text=d.text, source=d.source,
                             jurisdiction=d.jurisdiction, doc_type=d.doc_type))
    # SEC docs via full-text search
    sec = SECClient(user_agent=settings.sec_user_agent, cache_dir=Path("data/cache"))
    for hit in sec.full_text_search(args.sec_query, forms=["8-K", "10-K"], limit=args.sec_limit):
        docs.append(DocInput(doc_id=hit.accession_no or hit.title, title=hit.title,
                             text=f"{hit.title} {hit.form_type} filed {hit.filed_date}",
                             source="sec", jurisdiction="US-SEC", doc_type="filing",
                             form_type=hit.form_type, accession_no=hit.accession_no,
                             filed_date=hit.filed_date))
    n = ingest_documents(docs, store=store, dense=dense, sparse=sparse)
    print(f"Ingested {n} chunks from {len(docs)} documents.")


def cmd_query(args) -> None:
    settings = get_settings()
    store, dense, sparse = _build(settings)
    provider = OllamaProvider(host=settings.ollama_host, default_model=settings.ollama_chat_model)
    agent = RetrieverAgent(store=store, dense=dense, sparse=sparse,
                           provider=provider, rerank_model=settings.ollama_chat_model)
    filters = RetrievalFilters(jurisdiction=args.jurisdiction, source=args.source)
    for i, c in enumerate(agent.retrieve(args.query, filters=filters), 1):
        print(f"\n#{i} [{c.payload.get('source')}/{c.payload.get('doc_type')}] "
              f"{c.payload.get('title')}  (score={c.score:.3f})")
        print(f"   {c.text[:200]}...")
        if c.rerank_rationale:
            print(f"   why: {c.rerank_rationale}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="regintel")
    sub = parser.add_subparsers(required=True)
    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--sec-query", default="insider trading policy")
    p_ing.add_argument("--sec-limit", type=int, default=5)
    p_ing.set_defaults(func=cmd_ingest)
    p_q = sub.add_parser("query")
    p_q.add_argument("query")
    p_q.add_argument("--jurisdiction", default=None)
    p_q.add_argument("--source", default=None)
    p_q.set_defaults(func=cmd_query)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the full (non-live) suite**

Run: `uv run pytest -v`
Expected: ALL non-live tests PASS (live tests deselected).

- [ ] **Step 4: Run the live smoke test manually (requires Ollama running)**

Run: `uv run pytest tests/test_cli_live.py -m live -v`
Expected: PASS (proves bge-m3 + RRF + gpt-oss rerank work end-to-end).

- [ ] **Step 5: Commit**

```bash
git add src/regintel/cli.py tests/test_cli_live.py
git commit -m "feat: add ingest/query CLI and end-to-end live smoke test"
```

---

## Task 19: README + final verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Regulatory Intelligence System

Multi-agent RAG over **live SEC EDGAR filings** + synthetic internal compliance docs.
Phase 0+1: foundations + hybrid retrieval spine.

## Architecture
- **LLM layer**: pluggable provider (Ollama Cloud default, Claude optional), role-tiered.
- **Embeddings**: bge-m3 dense (via Ollama) + FastEmbed BM25 sparse.
- **Store**: Qdrant `corpus` collection, dense+sparse named vectors, server-side RRF fusion + payload filters.
- **Retrieval**: hybrid search → LLM listwise rerank (`RetrieverAgent`).

## Setup
1. `uv sync --extra dev`
2. `cp .env.example .env` and set `SEC_USER_AGENT` to "Your Name your@email".
3. Pull models: `ollama pull bge-m3` (gpt-oss:120b-cloud already available).
4. Start Qdrant: `docker compose up -d`  (or set `QDRANT_EMBEDDED=true` to skip Docker).

## Demo
```bash
uv run python -m regintel.cli ingest --sec-query "insider trading policy" --sec-limit 5
uv run python -m regintel.cli query "What are our obligations around insider trading?" --jurisdiction US-SEC
```

## Tests
- `uv run pytest`            # fast unit/integration (mocked)
- `uv run pytest -m live`    # requires Ollama + (embedded) Qdrant
```

- [ ] **Step 2: Final full verification**

Run: `uv run pytest -v && uv run ruff check src tests`
Expected: all non-live tests PASS; ruff clean (fix any lint).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and demo"
```

---

## Self-Review (completed by planner)

**Spec coverage:**
- Pluggable LLM layer (Ollama+Claude, role router) → Tasks 4-7 ✅
- Tiered routing defaults → Task 7 ✅
- Typed AgentState (full schema) → Task 3 ✅
- Shared `corpus` collection, metadata filters → Tasks 8-9 ✅
- bge-m3 dense via Ollama → Task 10 ✅
- FastEmbed BM25 sparse → Task 11 ✅
- Hybrid Query-API + RRF + payload/date filters → Task 9 ✅
- SEC EDGAR live fetch + cache + throttle + UA → Task 13 ✅
- Synthetic internal docs with deliberate gaps → Task 14 ✅
- Token-aware chunking w/ overlap + indices → Task 12 ✅
- Idempotent indexing (deterministic ids) → Tasks 8-9, 15 ✅
- LLM listwise reranker → Task 16 ✅
- RetrieverAgent (filters → hybrid → rerank) → Task 17 ✅
- Error handling (retries, LLMError, JSON fallback, embedded Qdrant fallback) → Tasks 5,9,10,13 ✅
- Testing strategy (unit, integration embedded Qdrant, recorded SEC fixture, mocked providers, `live` marker) → throughout ✅
- DoD #1-6 → Tasks 0,5,9,15,17,19 ✅

**Placeholder scan:** none — every code step shows complete code.

**Type consistency:** `ChunkRecord` (Task 9) fields used identically in Task 15; `RetrievedChunk`/`RetrievalFilters` (Task 2) consistent across store/rerank/retriever; `SparseVec` (Task 11) `.indices/.values` used consistently in pipeline/retriever; `point_id` signature stable (Tasks 8,9). Reranker schema keys (`ranking`/`index`/`rationale`) consistent between Task 16 impl and Task 17 fake.

**Note for executor:** Qdrant `query_points`/`Prefetch`/`FusionQuery` APIs assume `qdrant-client>=1.12`. If the installed client differs, adapt the fusion call per its docs — the behavior (dense+sparse prefetch, RRF, payload filter) is the contract to preserve.
```
