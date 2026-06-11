from regintel.rerank.llm_reranker import rerank
from regintel.types import RetrievalFilters, RetrievedChunk


class RetrieverAgent:
    """Hybrid retrieve (dense+sparse RRF) then LLM rerank."""

    def __init__(
        self, *, store, dense, sparse, provider, rerank_model: str,
        candidate_limit: int = 20, top_k: int = 8,
    ) -> None:
        self._store = store
        self._dense = dense
        self._sparse = sparse
        self._provider = provider
        self._rerank_model = rerank_model
        self._candidate_limit = candidate_limit
        self._top_k = top_k

    def retrieve(self, query: str, *, filters: RetrievalFilters | None = None) -> list[RetrievedChunk]:
        filters = filters or RetrievalFilters()
        dense_vec = self._dense.embed_one(query)
        sparse_vec = self._sparse.encode_query(query)
        candidates = self._store.hybrid_search(
            dense=dense_vec,
            sparse_indices=sparse_vec.indices,
            sparse_values=sparse_vec.values,
            filters=filters,
            limit=self._candidate_limit,
        )
        if not candidates:
            return []
        return rerank(
            query, candidates,
            provider=self._provider, model=self._rerank_model, top_k=self._top_k,
        )
