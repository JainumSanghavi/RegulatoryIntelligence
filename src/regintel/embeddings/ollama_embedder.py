import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from regintel.llm.base import LLMError


class OllamaEmbedder:
    """Dense embeddings via Ollama /api/embed (bge-m3)."""

    def __init__(self, host: str, model: str, *, timeout: float = 60.0, max_retries: int = 3) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

    def embed(self, texts: list[str]) -> list[list[float]]:
        @retry(
            retry=retry_if_exception_type((httpx.HTTPError,)),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.5, max=8),
            reraise=True,
        )
        def _do() -> list[list[float]]:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._host}/api/embed",
                    json={"model": self._model, "input": texts},
                )
                resp.raise_for_status()
                return resp.json()["embeddings"]

        try:
            return _do()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama embed failed: {exc}") from exc

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
