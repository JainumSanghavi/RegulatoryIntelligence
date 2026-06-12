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
        max_output_tokens: int = 8192,
    ) -> None:
        self._host = host.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        self._max_retries = max_retries
        # Generous output cap so large structured outputs (e.g. multi-finding
        # analyses) are not truncated into unparseable JSON.
        self._max_output_tokens = max_output_tokens

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
            "options": {"temperature": temperature, "num_predict": self._max_output_tokens},
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
        # Try up to twice: cloud models occasionally emit prose/partial JSON. On a
        # parse miss, reprompt once with a stricter correction before giving up.
        last_content = ""
        for attempt in range(2):
            attempt_msgs = msgs
            if attempt == 1:
                attempt_msgs = msgs + [{
                    "role": "user",
                    "content": "Your previous reply was not valid JSON. Output ONLY the JSON value now, nothing else.",
                }]
            payload = {
                "model": model or self._default_model,
                "messages": attempt_msgs,
                "stream": False,
                "format": schema,
                "options": {"temperature": temperature, "num_predict": self._max_output_tokens},
            }
            last_content = self._post_chat(payload)["message"]["content"]
            parsed = _parse_json(last_content)
            if parsed is not None:
                return parsed
        raise LLMError(f"Ollama returned non-JSON structured content: {last_content!r}")
