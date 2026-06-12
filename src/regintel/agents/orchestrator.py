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
