from regintel.ingest.sec_edgar import SECFiling
from regintel.ingest.sec_ingest import sec_query_to_docs


class _FakeClient:
    def __init__(self, filings, bodies):
        self._filings = filings
        self._bodies = bodies            # {url: text}
        self.fetched = []

    def full_text_search(self, query, *, forms=None, limit=10):
        return self._filings[:limit]

    def fetch_document(self, url):
        self.fetched.append(url)
        if url not in self._bodies:
            raise RuntimeError("boom")
        return self._bodies[url]


def test_sec_query_to_docs_fetches_bodies():
    filings = [
        SECFiling(accession_no="acc1", title="ACME 8-K", form_type="8-K",
                  filed_date="2026-05-01", cik="1", doc_url="http://x/acc1.htm"),
    ]
    client = _FakeClient(filings, {"http://x/acc1.htm": "Full filing body about insider trading."})
    docs = sec_query_to_docs(client, "insider trading", forms=["8-K"], limit=5)
    assert len(docs) == 1
    d = docs[0]
    assert d.source == "sec"
    assert d.jurisdiction == "US-SEC"
    assert d.doc_type == "filing"
    assert d.form_type == "8-K"
    assert d.accession_no == "acc1"
    assert d.url == "http://x/acc1.htm"
    assert "Full filing body about insider trading." in d.text
    assert "ACME 8-K" in d.text  # title is prepended for context


def test_sec_query_to_docs_skips_filings_without_url():
    filings = [
        SECFiling(accession_no="acc1", title="No URL", form_type="8-K",
                  filed_date="2026-05-01", cik=None, doc_url=None),
    ]
    client = _FakeClient(filings, {})
    docs = sec_query_to_docs(client, "q", forms=["8-K"], limit=5)
    assert docs == []
    assert client.fetched == []  # never attempted a fetch


def test_sec_query_to_docs_skips_unfetchable_bodies():
    filings = [
        SECFiling(accession_no="acc1", title="Good", form_type="8-K",
                  filed_date="2026-05-01", cik="1", doc_url="http://x/ok.htm"),
        SECFiling(accession_no="acc2", title="Bad", form_type="8-K",
                  filed_date="2026-05-02", cik="2", doc_url="http://x/bad.htm"),
    ]
    client = _FakeClient(filings, {"http://x/ok.htm": "body ok"})
    docs = sec_query_to_docs(client, "q", forms=["8-K"], limit=5)
    assert [d.accession_no for d in docs] == ["acc1"]  # acc2 raised, skipped
