import pytest


@pytest.mark.live
def test_ask_live_gap_check():
    """Requires Ollama (bge-m3 + gpt-oss) + embedded Qdrant. Uses gpt-oss for all roles."""
    from qdrant_client import QdrantClient

    from regintel.agents.retriever import RetrieverAgent
    from regintel.config import Settings
    from regintel.embeddings.ollama_embedder import OllamaEmbedder
    from regintel.embeddings.sparse import BM25Encoder
    from regintel.ingest.internal_docs import load_internal_docs
    from regintel.ingest.pipeline import DocInput, ingest_documents
    from regintel.llm.ollama_provider import OllamaProvider
    from regintel.orchestration.graph import build_default_graph, run_query
    from regintel.store.qdrant_store import QdrantStore
    from pathlib import Path

    s = Settings(_env_file=None)
    store = QdrantStore(client=QdrantClient(":memory:"))
    dense = OllamaEmbedder(host=s.ollama_host, model=s.ollama_embed_model)
    sparse = BM25Encoder()
    docs = [DocInput(doc_id=d.doc_id, title=d.title, text=d.text, source=d.source,
                     jurisdiction=d.jurisdiction, doc_type=d.doc_type)
            for d in load_internal_docs(Path("data/internal"))]
    docs.append(DocInput(doc_id="sec1", title="SEC Insider Trading",
                         text="Issuers must define quarterly blackout windows for insiders.",
                         source="sec", jurisdiction="US-SEC", doc_type="filing"))
    ingest_documents(docs, store=store, dense=dense, sparse=sparse)

    provider = OllamaProvider(host=s.ollama_host, default_model=s.ollama_chat_model)
    retriever = RetrieverAgent(store=store, dense=dense, sparse=sparse,
                               provider=provider, rerank_model=s.ollama_chat_model)
    graph = build_default_graph(s, retriever=retriever, provider=provider)
    report = run_query("Does our insider trading policy comply with SEC blackout rules?",
                       graph=graph)
    assert report.answer
    assert report.query_type is not None
    # Regression guard: the structured-output pipeline must actually parse JSON.
    # (Ollama Cloud ignores `format`; the provider embeds the schema in the prompt.)
    assert not any("non-JSON" in w for w in report.warnings), report.warnings
    # A real gap-check against a blackout-window doc should surface at least one finding.
    assert report.findings
