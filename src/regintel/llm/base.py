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
