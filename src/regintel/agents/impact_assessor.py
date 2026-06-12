from regintel.llm.base import ChatMessage
from regintel.types import Finding, Impact, RetrievedChunk

_VALID_SEVERITY = {"low", "medium", "high", "critical"}

_SCHEMA = {
    "type": "object",
    "properties": {
        "impacts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "affected_policies": {"type": "array", "items": {"type": "string"}},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "rationale": {"type": "string"},
                },
                "required": ["topic", "affected_policies", "severity", "rationale"],
            },
        }
    },
    "required": ["impacts"],
}

_SYSTEM = (
    "You assess the business impact of compliance gaps. For each gap, name which of the "
    "company's internal policies are affected (choose only from the provided policy titles), "
    "assign a severity (low, medium, high, critical), and give a one-sentence rationale."
)


class ImpactAssessor:
    def __init__(self, provider, model: str) -> None:
        self._provider = provider
        self._model = model

    def assess(self, findings: list[Finding], internal: list[RetrievedChunk]) -> list[Impact]:
        gaps = [f for f in findings if f.gap]
        if not gaps:
            return []
        known = {(c.payload or {}).get("title", "") for c in internal}
        gap_block = "\n".join(
            f"- {f.topic}: requires '{f.requirement}'; internal status: {f.internal_status}"
            for f in gaps
        )
        policy_block = "\n".join(f"- {t}" for t in sorted(known) if t) or "(none)"
        user = f"GAPS:\n{gap_block}\n\nINTERNAL POLICY TITLES:\n{policy_block}"
        out = self._provider.chat_structured(
            [ChatMessage("system", _SYSTEM), ChatMessage("user", user)],
            schema=_SCHEMA, model=self._model,
        )
        impacts: list[Impact] = []
        for im in out.get("impacts", []):
            severity = im.get("severity", "medium")
            if severity not in _VALID_SEVERITY:
                severity = "medium"
            policies = [p for p in im.get("affected_policies", []) if p in known]
            impacts.append(Impact(
                topic=im.get("topic", ""),
                affected_policies=policies,
                severity=severity,
                rationale=im.get("rationale", ""),
            ))
        return impacts
