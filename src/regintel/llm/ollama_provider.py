import json
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from regintel.llm.base import ChatMessage, LLMError, _normalize


def _extract_balanced(text: str) -> str | None:
    """Return the first balanced {...} or [...] JSON value found in text, or None."""
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
    return None


def _parse_json(content: str) -> Any | None:
    """Best-effort parse of a JSON value from model output (fences/prose tolerated)."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    snippet = _extract_balanced(text)
    if snippet is not None:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    return None


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
        # Ollama Cloud models do NOT enforce the `format` JSON-schema, so we also
        # embed the schema + a JSON-only instruction directly in the prompt. We
        # keep `format` too (local models honor it; harmless on cloud).
        msgs = _normalize(messages)
        msgs.append({
            "role": "user",
            "content": (
                "Respond with ONLY a single JSON value — no prose, no explanation, "
                "no markdown code fences — that matches EXACTLY this JSON schema:\n"
                + json.dumps(schema)
            ),
        })
        payload = {
            "model": model or self._default_model,
            "messages": msgs,
            "stream": False,
            "format": schema,
            "options": {"temperature": temperature},
        }
        data = self._post_chat(payload)
        content = data["message"]["content"]
        parsed = _parse_json(content)
        if parsed is None:
            raise LLMError(f"Ollama returned non-JSON structured content: {content!r}")
        return parsed
