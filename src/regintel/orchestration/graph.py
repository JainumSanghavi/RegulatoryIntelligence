from langgraph.graph import END, START, StateGraph

from regintel.config import Settings, get_settings
from regintel.llm.router import Role, resolve_model
from regintel.orchestration.nodes import (
    make_analyze_node, make_assess_node, make_classify_node, make_evaluate_node,
    make_report_node, make_retrieve_internal_node, make_retrieve_regulations_node,
    route_after_analyze, route_after_regulations,
)
from regintel.state import AgentState, new_state
from regintel.types import Report


def build_graph(*, retriever, orchestrator, analyst, assessor, reporter, evaluator):
    g = StateGraph(AgentState)
    g.add_node("classify", make_classify_node(orchestrator))
    g.add_node("retrieve_regulations", make_retrieve_regulations_node(retriever))
    g.add_node("retrieve_internal", make_retrieve_internal_node(retriever))
    g.add_node("analyze", make_analyze_node(analyst))
    g.add_node("assess", make_assess_node(assessor))
    g.add_node("report", make_report_node(reporter))
    g.add_node("evaluate", make_evaluate_node(evaluator))

    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve_regulations")
    g.add_conditional_edges("retrieve_regulations", route_after_regulations,
                            {"report": "report", "retrieve_internal": "retrieve_internal"})
    g.add_edge("retrieve_internal", "analyze")
    g.add_conditional_edges("analyze", route_after_analyze,
                            {"assess": "assess", "report": "report"})
    g.add_edge("assess", "report")
    g.add_edge("report", "evaluate")
    g.add_edge("evaluate", END)
    return g.compile()


def build_default_graph(settings: Settings | None = None, *, retriever=None, provider=None):
    """Wire the real agents using Ollama + the existing retrieval stack."""
    settings = settings or get_settings()
    from regintel.agents.analyst import Analyst
    from regintel.agents.impact_assessor import ImpactAssessor
    from regintel.agents.orchestrator import Orchestrator
    from regintel.agents.reporter import Reporter
    from regintel.llm.ollama_provider import OllamaProvider

    if provider is None:
        provider = OllamaProvider(host=settings.ollama_host,
                                  default_model=settings.ollama_chat_model)
    if retriever is None:
        from qdrant_client import QdrantClient
        from regintel.agents.retriever import RetrieverAgent
        from regintel.embeddings.ollama_embedder import OllamaEmbedder
        from regintel.embeddings.sparse import BM25Encoder
        from regintel.store.qdrant_store import QdrantStore
        client = (QdrantClient(path="./qdrant_storage") if settings.qdrant_embedded
                  else QdrantClient(url=settings.qdrant_url))
        retriever = RetrieverAgent(
            store=QdrantStore(client=client),
            dense=OllamaEmbedder(host=settings.ollama_host, model=settings.ollama_embed_model),
            sparse=BM25Encoder(),
            provider=provider, rerank_model=settings.ollama_chat_model,
        )
    from regintel.agents.evaluator import Evaluator
    chat = resolve_model(Role.ANALYST, settings)
    frontier = resolve_model(Role.IMPACT_ASSESSOR, settings)
    return build_graph(
        retriever=retriever,
        orchestrator=Orchestrator(provider, model=resolve_model(Role.ORCHESTRATOR, settings)),
        analyst=Analyst(provider, model=chat),
        assessor=ImpactAssessor(provider, model=frontier),
        reporter=Reporter(provider, model=resolve_model(Role.REPORTER, settings)),
        evaluator=Evaluator(provider, model=resolve_model(Role.EVALUATOR, settings)),
    )


def run_query(query: str, *, graph) -> Report:
    final = graph.invoke(new_state(query))
    return final["report"]
