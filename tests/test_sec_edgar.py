from pathlib import Path

import httpx
import respx

from regintel.ingest.sec_edgar import SECClient, SECFiling


def test_html_to_text_strips_markup():
    html = Path("tests/fixtures/sec_filing.html").read_text()
    text = SECClient.html_to_text(html)
    assert "Risk Factors" in text
    assert "insider trading policy" in text
    assert "ignore me" not in text
    assert "<p>" not in text


@respx.mock
def test_fetch_document_uses_user_agent_and_caches(tmp_path):
    url = "https://www.sec.gov/Archives/edgar/data/1/x.htm"
    route = respx.get(url).mock(
        return_value=httpx.Response(200, text="<html><body><p>Hello SEC</p></body></html>")
    )
    client = SECClient(user_agent="Tester test@example.com", cache_dir=tmp_path)
    text1 = client.fetch_document(url)
    assert "Hello SEC" in text1
    assert route.calls.last.request.headers["user-agent"] == "Tester test@example.com"
    text2 = client.fetch_document(url)
    assert text2 == text1
    assert route.call_count == 1


@respx.mock
def test_full_text_search_parses_hits():
    respx.get(url__startswith="https://efts.sec.gov/LATEST/search-index").mock(
        return_value=httpx.Response(200, json={
            "hits": {"hits": [
                {"_id": "0001-24-000001:doc.htm",
                 "_source": {"display_names": ["ACME (CIK 0000001)"],
                             "form": "8-K", "file_date": "2026-05-01"}}
            ]}
        })
    )
    client = SECClient(user_agent="Tester test@example.com")
    hits = client.full_text_search("insider trading", forms=["8-K"], limit=1)
    assert len(hits) == 1
    assert isinstance(hits[0], SECFiling)
    assert hits[0].form_type == "8-K"
    assert hits[0].filed_date == "2026-05-01"
    # No `ciks` in source -> cannot build a document URL; degrade gracefully.
    assert hits[0].doc_url is None


def test_build_doc_url_strips_dashes_and_leading_zeros():
    url = SECClient.build_doc_url(
        cik="0001554225", accession_no="0001604232-14-000006", filename="ex10_7.htm"
    )
    assert url == "https://www.sec.gov/Archives/edgar/data/1554225/000160423214000006/ex10_7.htm"


@respx.mock
def test_full_text_search_builds_doc_url_from_ciks():
    respx.get(url__startswith="https://efts.sec.gov/LATEST/search-index").mock(
        return_value=httpx.Response(200, json={
            "hits": {"hits": [
                {"_id": "0001604232-14-000006:ex10_7.htm",
                 "_source": {"display_names": ["Pladeo Corp. (CIK 0001554225)"],
                             "ciks": ["0001554225"], "form": "8-K", "file_date": "2014-04-04"}}
            ]}
        })
    )
    client = SECClient(user_agent="Tester test@example.com")
    hit = client.full_text_search("insider trading", forms=["8-K"], limit=1)[0]
    assert hit.cik == "0001554225"
    assert hit.doc_url == (
        "https://www.sec.gov/Archives/edgar/data/1554225/000160423214000006/ex10_7.htm"
    )
