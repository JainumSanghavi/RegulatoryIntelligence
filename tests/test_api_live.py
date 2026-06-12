import pytest


@pytest.mark.live
def test_ask_endpoint_live():
    """Requires Ollama + (embedded) Qdrant with ingested data. Uses default wiring."""
    from fastapi.testclient import TestClient
    from regintel.api.app import create_app

    client = TestClient(create_app())
    # Hit /changelog first, THEN /ask: in embedded Qdrant mode both endpoints must
    # share one client (a second client on the same path raises). Regression guard.
    assert client.get("/changelog?limit=5").status_code == 200
    r = client.post("/ask", json={"query": "What are insider trading blackout window rules?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["query_type"] in {"lookup", "gap_check", "impact"}
    assert body["eval"] is None or 0.0 <= body["eval"]["faithfulness"] <= 1.0
