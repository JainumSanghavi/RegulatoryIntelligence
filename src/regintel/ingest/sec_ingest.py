"""Bridge: turn a SEC full-text-search query into ingestible documents.

Fetches each matching filing's actual body (not just metadata) and wraps it
as a `DocInput`. Filings without a resolvable document URL, or whose body
cannot be fetched, are skipped so one bad filing never fails the whole ingest.
"""

import logging

from regintel.ingest.pipeline import DocInput

logger = logging.getLogger(__name__)


def sec_query_to_docs(client, query: str, *, forms: list[str] | None = None, limit: int = 10) -> list[DocInput]:
    docs: list[DocInput] = []
    for filing in client.full_text_search(query, forms=forms, limit=limit):
        if not filing.doc_url:
            logger.warning("Skipping %s: no resolvable document URL", filing.accession_no)
            continue
        try:
            body = client.fetch_document(filing.doc_url)
        except Exception as exc:  # noqa: BLE001 - skip unfetchable filings, don't fail ingest
            logger.warning("Skipping %s: fetch failed (%s)", filing.accession_no, exc)
            continue
        text = f"{filing.title}\n\n{body}"
        docs.append(
            DocInput(
                doc_id=filing.accession_no,
                title=filing.title,
                text=text,
                source="sec",
                jurisdiction="US-SEC",
                doc_type="filing",
                url=filing.doc_url,
                form_type=filing.form_type,
                accession_no=filing.accession_no,
                filed_date=filing.filed_date,
            )
        )
    return docs
