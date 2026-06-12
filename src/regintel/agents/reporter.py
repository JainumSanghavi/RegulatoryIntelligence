from regintel.llm.base import ChatMessage
from regintel.types import Citation, Finding, Impact, QueryType, Report, RetrievedChunk, cite

_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited_indices": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "cited_indices"],
}

_SYSTEM = (
    "You are a compliance report writer. Using ONLY the numbered citations provided, write a "
    "clear, well-structured answer to the question. Insert inline markers like [0], [1] where "
    "each claim is supported, and return the list of citation indices you used. Do not invent "
    "facts beyond the citations."
)


def _dedupe(citations: list[Citation]) -> list[Citation]:
    seen: set[tuple[str, int]] = set()
    out: list[Citation] = []
    for c in citations:
        key = (c.doc_id, c.chunk_index)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


class Reporter:
    def __init__(self, provider, model: str) -> None:
        self._provider = provider
        self._model = model

    def _pool(self, findings: list[Finding], regulations: list[RetrievedChunk]) -> list[Citation]:
        pool: list[Citation] = []
        for f in findings:
            pool.extend(f.citations)
        pool.extend(cite(ch) for ch in regulations)
        return _dedupe(pool)

    def report(
        self,
        query: str,
        query_type: QueryType,
        findings: list[Finding],
        impacts: list[Impact],
        regulations: list[RetrievedChunk],
        internal: list[RetrievedChunk],
    ) -> Report:
        pool = self._pool(findings, regulations)
        if not pool:
            return Report(
                query_type=query_type,
                answer="No relevant regulations found for this question.",
            )
        numbered = "\n".join(
            f"[{i}] ({c.source}) {c.title}: {c.quote}" for i, c in enumerate(pool)
        )
        summary = ""
        if findings:
            summary += "\nFINDINGS:\n" + "\n".join(
                f"- {f.topic}: {'GAP' if f.gap else 'ok'} — {f.explanation}" for f in findings
            )
        if impacts:
            summary += "\nIMPACTS:\n" + "\n".join(
                f"- {im.topic}: severity={im.severity}; policies={im.affected_policies}"
                for im in impacts
            )
        user = (
            f"Question: {query}\nQuery type: {query_type.value}{summary}\n\nCITATIONS:\n{numbered}"
        )
        out = self._provider.chat_structured(
            [ChatMessage("system", _SYSTEM), ChatMessage("user", user)],
            schema=_SCHEMA,
            model=self._model,
        )
        cited = [
            pool[i]
            for i in out.get("cited_indices", [])
            if isinstance(i, int) and 0 <= i < len(pool)
        ]
        return Report(
            query_type=query_type,
            answer=out.get("answer", ""),
            citations=cited,
            findings=findings,
            impacts=impacts,
        )
