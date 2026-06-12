from regintel.llm.base import ChatMessage, LLMError
from regintel.types import Citation, EvalScores, Report

FAITHFULNESS_THRESHOLD = 0.7

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "has_citation": {"type": "boolean"},
                },
                "required": ["claim", "supported", "has_citation"],
            },
        }
    },
    "required": ["claims"],
}

_CONFLICTS_SCHEMA = {
    "type": "object",
    "properties": {
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            },
        }
    },
    "required": ["conflicts"],
}

_CLAIMS_SYSTEM = (
    "You are a strict faithfulness judge. Decompose the ANSWER into atomic factual claims. "
    "For each claim decide: supported = is it supported by the numbered CITATIONS (and only "
    "those, not outside knowledge); has_citation = does the claim carry an inline marker like "
    "[0]. Be conservative: if a claim is not clearly supported by the citations, mark it unsupported."
)

_CONFLICTS_SYSTEM = (
    "You detect material contradictions between source passages. Given the numbered passages, "
    "list any pair that materially contradict each other, each as a short description. "
    "If there are none, return an empty list."
)


def _number(citations: list[Citation]) -> str:
    return "\n".join(f"[{i}] ({c.source}) {c.title}: {c.quote}" for i, c in enumerate(citations))


class Evaluator:
    def __init__(self, provider, model: str) -> None:
        self._provider = provider
        self._model = model

    def evaluate(self, query: str, report: Report) -> EvalScores:
        answer = (report.answer or "").strip()
        if not report.citations or answer.lower().startswith("no relevant regulations"):
            return EvalScores(1.0, 1.0, [], flagged=False, notes="no content to evaluate")

        cites = _number(report.citations)

        try:
            out = self._provider.chat_structured(
                [ChatMessage("system", _CLAIMS_SYSTEM),
                 ChatMessage("user", f"ANSWER:\n{report.answer}\n\nCITATIONS:\n{cites}")],
                schema=_CLAIMS_SCHEMA, model=self._model,
            )
            claims = out.get("claims", [])
        except LLMError as exc:
            return EvalScores(0.0, 0.0, [], flagged=True, notes=f"evaluation failed: {exc}")

        if claims:
            supported = sum(1 for c in claims if c.get("supported"))
            cited = sum(1 for c in claims if c.get("has_citation"))
            faithfulness = supported / len(claims)
            citation_coverage = cited / len(claims)
        else:
            faithfulness = 1.0
            citation_coverage = 1.0

        conflicts: list[str] = []
        conflict_note = ""
        try:
            cout = self._provider.chat_structured(
                [ChatMessage("system", _CONFLICTS_SYSTEM),
                 ChatMessage("user", f"PASSAGES:\n{cites}")],
                schema=_CONFLICTS_SCHEMA, model=self._model,
            )
            conflicts = [c.get("description", "") for c in cout.get("conflicts", []) if c.get("description")]
        except LLMError as exc:
            conflict_note = f" (conflict check skipped: {exc})"

        flagged = faithfulness < FAITHFULNESS_THRESHOLD or bool(conflicts)
        notes = ("ok" if not flagged else "low faithfulness or conflicts detected") + conflict_note
        return EvalScores(
            faithfulness=round(faithfulness, 3),
            citation_coverage=round(citation_coverage, 3),
            conflicts=conflicts,
            flagged=flagged,
            notes=notes,
        )
