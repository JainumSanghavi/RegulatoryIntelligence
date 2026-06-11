import pytest

from regintel.config import Settings


@pytest.mark.live
def test_end_to_end_ingest_and_query():
    """Requires Ollama (bge-m3 + gpt-oss) running and Qdrant embedded."""
    from qdrant_client import QdrantClient
    from regintel.store.qdrant_store import QdrantStore
    from regintel.embeddings.ollama_embedder import OllamaEmbedder
    from regintel.embeddings.sparse import BM25Encoder
    from regintel.ingest.pipeline import DocInput, ingest_documents
    from regintel.agents.retriever import RetrieverAgent
    from regintel.llm.ollama_provider import OllamaProvider
    from regintel.types import RetrievalFilters

    s = Settings(_env_file=None)
    store = QdrantStore(client=QdrantClient(":memory:"))
    dense = OllamaEmbedder(host=s.ollama_host, model=s.ollama_embed_model)
    sparse = BM25Encoder()
    docs = [
        DocInput(doc_id="d1", title="Insider Trading", source="sec", jurisdiction="US-SEC",
                 doc_type="filing", text="Insider trading on material non-public information is prohibited."),
        DocInput(doc_id="d2", title="Office Supplies", source="sec", jurisdiction="US-SEC",
                 doc_type="filing", text="We purchase staplers and paper for the office."),
    ]
    n = ingest_documents(docs, store=store, dense=dense, sparse=sparse)
    assert n >= 2
    provider = OllamaProvider(host=s.ollama_host, default_model=s.ollama_chat_model)
    agent = RetrieverAgent(store=store, dense=dense, sparse=sparse,
                           provider=provider, rerank_model=s.ollama_chat_model, top_k=1)
    out = agent.retrieve("rules about trading on insider information",
                         filters=RetrievalFilters(jurisdiction="US-SEC"))
    assert out
    assert out[0].doc_id == "d1"
