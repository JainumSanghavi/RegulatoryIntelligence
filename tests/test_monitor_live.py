import pytest


@pytest.mark.live
def test_monitor_poll_live():
    """Requires Ollama (bge-m3 + gpt-oss) + network to SEC. Uses embedded Qdrant."""
    from qdrant_client import QdrantClient
    from regintel.config import Settings
    from regintel.monitoring.scheduler import build_default_monitor

    s = Settings(_env_file=None)
    monitor = build_default_monitor(s, client=QdrantClient(":memory:"))
    first = monitor.poll("insider trading policy", forms=["8-K"], limit=2)
    assert isinstance(first, list)
    assert len(first) >= 1
    assert all(e.summary for e in first)
    second = monitor.poll("insider trading policy", forms=["8-K"], limit=2)
    assert second == []
