from regintel.llm.base import ChatMessage
from regintel.types import RetrievedChunk

_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["index", "rationale"],
            },
        }
    },
    "required": ["ranking"],
}

_SYSTEM = (
    "You are a search reranker. Given a query and numbered candidate passages, "
    "return them ordered most-relevant first. Use only the provided passages."
)


def _format_candidates(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks):
        snippet = c.text[:600].replace("\n", " ")
        lines.append(f"[{i}] {snippet}")
    return "\n".join(lines)


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    provider,
    model: str,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    user = (
        f"Query: {query}\n\nCandidates:\n{_format_candidates(chunks)}\n\n"
        f"Return a JSON ranking (best first) of the candidate indices with a short rationale each."
    )
    result = provider.chat_structured(
        [ChatMessage("system", _SYSTEM), ChatMessage("user", user)],
        schema=_RERANK_SCHEMA, model=model,
    )
    # Tolerate models returning a bare list instead of {"ranking": [...]}.
    ranking_items = result if isinstance(result, list) else result.get("ranking", [])
    ordered: list[RetrievedChunk] = []
    seen: set[int] = set()
    for item in ranking_items:
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(chunks) or idx in seen:
            continue
        seen.add(idx)
        chunk = chunks[idx]
        chunk.rerank_rationale = item.get("rationale")
        ordered.append(chunk)
    # Safety net: if the model returned no usable ranking (empty/garbage),
    # degrade gracefully to the original hybrid order rather than dropping
    # every result. Chunks the model explicitly omitted are still dropped.
    if not ordered:
        return chunks[:top_k]
    return ordered[:top_k]
