import argparse
from pathlib import Path

from qdrant_client import QdrantClient

from regintel.agents.retriever import RetrieverAgent
from regintel.config import get_settings
from regintel.embeddings.ollama_embedder import OllamaEmbedder
from regintel.embeddings.sparse import BM25Encoder
from regintel.ingest.internal_docs import load_internal_docs
from regintel.ingest.pipeline import DocInput, ingest_documents
from regintel.ingest.sec_edgar import SECClient
from regintel.llm.ollama_provider import OllamaProvider
from regintel.store.qdrant_store import QdrantStore
from regintel.types import RetrievalFilters


def _build(settings):
    if settings.qdrant_embedded:
        client = QdrantClient(path="./qdrant_storage")
    else:
        client = QdrantClient(url=settings.qdrant_url)
    store = QdrantStore(client=client)
    dense = OllamaEmbedder(host=settings.ollama_host, model=settings.ollama_embed_model)
    sparse = BM25Encoder()
    return store, dense, sparse


def cmd_ingest(args) -> None:
    settings = get_settings()
    store, dense, sparse = _build(settings)
    docs: list[DocInput] = []
    for d in load_internal_docs(Path("data/internal")):
        docs.append(DocInput(doc_id=d.doc_id, title=d.title, text=d.text, source=d.source,
                             jurisdiction=d.jurisdiction, doc_type=d.doc_type))
    sec = SECClient(user_agent=settings.sec_user_agent, cache_dir=Path("data/cache"))
    for hit in sec.full_text_search(args.sec_query, forms=["8-K", "10-K"], limit=args.sec_limit):
        docs.append(DocInput(doc_id=hit.accession_no or hit.title, title=hit.title,
                             text=f"{hit.title} {hit.form_type} filed {hit.filed_date}",
                             source="sec", jurisdiction="US-SEC", doc_type="filing",
                             form_type=hit.form_type, accession_no=hit.accession_no,
                             filed_date=hit.filed_date))
    n = ingest_documents(docs, store=store, dense=dense, sparse=sparse)
    print(f"Ingested {n} chunks from {len(docs)} documents.")


def cmd_query(args) -> None:
    settings = get_settings()
    store, dense, sparse = _build(settings)
    provider = OllamaProvider(host=settings.ollama_host, default_model=settings.ollama_chat_model)
    agent = RetrieverAgent(store=store, dense=dense, sparse=sparse,
                           provider=provider, rerank_model=settings.ollama_chat_model)
    filters = RetrievalFilters(jurisdiction=args.jurisdiction, source=args.source)
    for i, c in enumerate(agent.retrieve(args.query, filters=filters), 1):
        print(f"\n#{i} [{c.payload.get('source')}/{c.payload.get('doc_type')}] "
              f"{c.payload.get('title')}  (score={c.score:.3f})")
        print(f"   {c.text[:200]}...")
        if c.rerank_rationale:
            print(f"   why: {c.rerank_rationale}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="regintel")
    sub = parser.add_subparsers(required=True)
    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--sec-query", default="insider trading policy")
    p_ing.add_argument("--sec-limit", type=int, default=5)
    p_ing.set_defaults(func=cmd_ingest)
    p_q = sub.add_parser("query")
    p_q.add_argument("query")
    p_q.add_argument("--jurisdiction", default=None)
    p_q.add_argument("--source", default=None)
    p_q.set_defaults(func=cmd_query)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
