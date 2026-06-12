from regintel.llm.base import ChatMessage
from regintel.types import Citation, Finding, RetrievedChunk, cite

_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "requirement": {"type": "string"},
                    "internal_status": {"type": "string"},
                    "gap": {"type": "boolean"},
                    "explanation": {"type": "string"},
                    "regulation_refs": {"type": "array", "items": {"type": "integer"}},
                    "internal_refs": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["topic", "requirement", "internal_status", "gap", "explanation"],
            },
        }
    },
    "required": ["findings"],
}

_SYSTEM = (
    "You are a compliance analyst. Compare what the REGULATIONS require against what the "
    "company's INTERNAL DOCUMENTS say. For each relevant topic, state the requirement, the "
    "internal status (or 'absent'), whether there is a gap, and a short explanation. "
    "Cite supporting passages by their integer index using regulation_refs and internal_refs. "
    "Use only the provided passages."
)


def _number(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(none)"
    return "\n".join(f"[{i}] {c.text[:600]}".replace("\n", " ") for i, c in enumerate(chunks))


def _resolve(indices, chunks: list[RetrievedChunk]) -> list[Citation]:
    out: list[Citation] = []
    for idx in indices or []:
        if isinstance(idx, int) and 0 <= idx < len(chunks):
            out.append(cite(chunks[idx]))
    return out


class Analyst:
    def __init__(self, provider, model: str) -> None:
        self._provider = provider
        self._model = model

    def analyze(self, query: str, regulations: list[RetrievedChunk],
                internal: list[RetrievedChunk]) -> list[Finding]:
        if not regulations:
            return []
        user = (
            f"Question: {query}\n\nREGULATIONS:\n{_number(regulations)}\n\n"
            f"INTERNAL DOCUMENTS:\n{_number(internal)}"
        )
        out = self._provider.chat_structured(
            [ChatMessage("system", _SYSTEM), ChatMessage("user", user)],
            schema=_SCHEMA, model=self._model,
        )
        findings: list[Finding] = []
        for f in out.get("findings", []):
            citations = _resolve(f.get("regulation_refs"), regulations) + \
                _resolve(f.get("internal_refs"), internal)
            findings.append(Finding(
                topic=f.get("topic", ""),
                requirement=f.get("requirement", ""),
                internal_status=f.get("internal_status", ""),
                gap=bool(f.get("gap", False)),
                explanation=f.get("explanation", ""),
                citations=citations,
            ))
        return findings
