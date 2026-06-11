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
