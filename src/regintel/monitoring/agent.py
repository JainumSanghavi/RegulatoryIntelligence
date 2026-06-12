import logging
from datetime import datetime, timezone

from regintel.ingest.pipeline import DocInput, ingest_documents
from regintel.llm.base import ChatMessage
from regintel.types import ChangelogEntry

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = (
    "Summarize in ONE concise sentence what this SEC filing is about and why it might "
    "matter for compliance. No preamble."
)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MonitorAgent:
    def __init__(self, *, sec_client, corpus_store, dense, sparse, changelog_store,
                 provider, summary_model: str) -> None:
        self._sec = sec_client
        self._corpus = corpus_store
        self._dense = dense
        self._sparse = sparse
        self._changelog = changelog_store
        self._provider = provider
        self._summary_model = summary_model

    def poll(self, query: str, *, forms: list[str], limit: int = 10) -> list[ChangelogEntry]:
        filings = self._sec.full_text_search(query, forms=forms, limit=limit)
        recorded: list[ChangelogEntry] = []
        for f in filings:
            if not f.doc_url or self._changelog.is_seen(f.accession_no):
                continue
            try:
                body = self._sec.fetch_document(f.doc_url)
                doc = DocInput(
                    doc_id=f.accession_no, title=f.title, text=f"{f.title}\n\n{body}",
                    source="sec", jurisdiction="US-SEC", doc_type="filing",
                    url=f.doc_url, form_type=f.form_type, accession_no=f.accession_no,
                    filed_date=f.filed_date,
                )
                ingest_documents([doc], store=self._corpus, dense=self._dense, sparse=self._sparse)
                summary = self._summarize(f.title, body)
                vector = self._dense.embed_one(summary or f.title)
                entry = ChangelogEntry(
                    accession_no=f.accession_no, title=f.title, form_type=f.form_type,
                    filed_date=f.filed_date, url=f.doc_url, summary=summary,
                    detected_at=now_utc_iso(),
                )
                self._changelog.record(entry, vector)
                recorded.append(entry)
            except Exception as exc:  # noqa: BLE001 - skip a bad filing, keep polling
                logger.warning("Monitor skipping %s: %s", f.accession_no, exc)
                continue
        return recorded

    def _summarize(self, title: str, body: str) -> str:
        try:
            text = self._provider.chat(
                [ChatMessage("system", _SUMMARY_SYSTEM),
                 ChatMessage("user", f"{title}\n\n{body[:2000]}")],
                model=self._summary_model,
            )
            return text.strip()
        except Exception as exc:  # noqa: BLE001 - summary is non-critical
            logger.warning("summary failed for %s: %s", title, exc)
            return title
