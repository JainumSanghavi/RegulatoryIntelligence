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
