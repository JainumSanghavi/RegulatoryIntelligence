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
