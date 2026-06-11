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
