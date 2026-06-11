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
            "format": schema,
            "options": {"temperature": temperature},
        }
        data = self._post_chat(payload)
        content = data["message"]["content"]
        # Strip markdown code fences that some models emit despite `format` being set.
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # Drop opening fence (```json or ```) and closing fence (```)
            inner = lines[1:] if lines[0].startswith("```") else lines
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            stripped = "\n".join(inner)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Ollama returned non-JSON structured content: {content!r}") from exc
