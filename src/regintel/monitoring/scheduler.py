import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from regintel.config import Settings, get_settings

logger = logging.getLogger(__name__)


def make_poll_job(monitor, *, query: str, forms: list[str], limit: int):
    def job() -> None:
        try:
            entries = monitor.poll(query, forms=forms, limit=limit)
            logger.info("Monitor poll: %d new filing(s)", len(entries))
        except Exception as exc:  # noqa: BLE001 - never kill the scheduler
            logger.error("Monitor poll failed: %s", exc)
    return job


def run_scheduler(monitor, *, query: str, forms: list[str], limit: int,
                  interval_seconds: int) -> None:
    job = make_poll_job(monitor, query=query, forms=forms, limit=limit)
    job()  # run once immediately on start
    scheduler = BlockingScheduler()
    scheduler.add_job(job, "interval", seconds=interval_seconds)
    logger.info("Monitor scheduler started (every %ds)", interval_seconds)
    scheduler.start()


def build_default_monitor(settings: Settings | None = None, *, client=None):
    settings = settings or get_settings()
    from pathlib import Path

    from qdrant_client import QdrantClient

    from regintel.embeddings.ollama_embedder import OllamaEmbedder
    from regintel.embeddings.sparse import BM25Encoder
    from regintel.ingest.sec_edgar import SECClient
    from regintel.llm.ollama_provider import OllamaProvider
    from regintel.monitoring.agent import MonitorAgent
    from regintel.store.changelog_store import ChangelogStore
    from regintel.store.qdrant_store import QdrantStore

    if client is None:
        client = (QdrantClient(path="./qdrant_storage") if settings.qdrant_embedded
                  else QdrantClient(url=settings.qdrant_url))
    corpus = QdrantStore(client=client)
    corpus.ensure_collection()
    changelog = ChangelogStore(client=client)
    changelog.ensure_collection()
    return MonitorAgent(
        sec_client=SECClient(user_agent=settings.sec_user_agent, cache_dir=Path("data/cache")),
        corpus_store=corpus,
        dense=OllamaEmbedder(host=settings.ollama_host, model=settings.ollama_embed_model),
        sparse=BM25Encoder(),
        changelog_store=changelog,
        provider=OllamaProvider(host=settings.ollama_host, default_model=settings.ollama_chat_model),
        summary_model=settings.ollama_chat_model,
    )
