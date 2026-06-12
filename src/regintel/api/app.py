from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from regintel.types import ChangelogEntry, Citation, Report

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX = _STATIC_DIR / "index.html"


class AskRequest(BaseModel):
    query: str


def _serialize_citation(c: Citation) -> dict:
    return {
        "doc_id": c.doc_id, "chunk_index": c.chunk_index, "title": c.title,
        "source": c.source, "url": c.url, "quote": c.quote,
    }


def _serialize_report(r: Report) -> dict:
    return {
        "query_type": r.query_type.value,
        "answer": r.answer,
        "citations": [_serialize_citation(c) for c in r.citations],
        "findings": [
            {
                "topic": f.topic, "requirement": f.requirement,
                "internal_status": f.internal_status, "gap": f.gap,
                "explanation": f.explanation,
                "citations": [_serialize_citation(c) for c in f.citations],
            }
            for f in r.findings
        ],
        "impacts": [
            {
                "topic": i.topic, "affected_policies": i.affected_policies,
                "severity": i.severity, "rationale": i.rationale,
            }
            for i in r.impacts
        ],
        "eval": (
            {
                "faithfulness": r.eval.faithfulness,
                "citation_coverage": r.eval.citation_coverage,
                "conflicts": r.eval.conflicts,
                "flagged": r.eval.flagged,
                "notes": r.eval.notes,
            }
            if r.eval is not None else None
        ),
        "warnings": r.warnings,
    }


def _serialize_changelog(e: ChangelogEntry) -> dict:
    return {
        "accession_no": e.accession_no, "title": e.title, "form_type": e.form_type,
        "filed_date": e.filed_date, "url": e.url, "summary": e.summary,
        "detected_at": e.detected_at,
    }


def create_app(*, graph=None, changelog_store=None) -> FastAPI:
    app = FastAPI(title="Regulatory Intelligence")
    state: dict = {"graph": graph, "changelog": changelog_store}

    def _get_graph():
        if state["graph"] is None:
            from regintel.orchestration.graph import build_default_graph
            state["graph"] = build_default_graph()
        return state["graph"]

    def _get_changelog():
        if state["changelog"] is None:
            from qdrant_client import QdrantClient
            from regintel.config import get_settings
            from regintel.store.changelog_store import ChangelogStore
            s = get_settings()
            client = (QdrantClient(path="./qdrant_storage") if s.qdrant_embedded
                      else QdrantClient(url=s.qdrant_url))
            store = ChangelogStore(client=client)
            store.ensure_collection()
            state["changelog"] = store
        return state["changelog"]

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        query = req.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        from regintel.orchestration.graph import run_query
        try:
            report = run_query(query, graph=_get_graph())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _serialize_report(report)

    @app.get("/changelog")
    def changelog(limit: int = 20) -> list[dict]:
        return [_serialize_changelog(e) for e in _get_changelog().list_recent(limit)]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_INDEX)

    return app


app = create_app()
