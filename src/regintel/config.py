from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    ollama_chat_model: str = "gpt-oss:120b-cloud"
    ollama_embed_model: str = "bge-m3"
    ollama_frontier_model: str = "kimi-k2.6:cloud"

    anthropic_api_key: str | None = None

    sec_user_agent: str = "RegIntel Example example@example.com"

    qdrant_url: str = "http://localhost:6333"
    qdrant_embedded: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
