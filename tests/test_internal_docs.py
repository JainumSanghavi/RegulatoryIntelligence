from pathlib import Path

from regintel.ingest.internal_docs import load_internal_docs


def test_load_internal_docs_reads_corpus():
    docs = load_internal_docs(Path("data/internal"))
    titles = {d.doc_id for d in docs}
    assert "insider_trading_policy" in titles
    assert len(docs) == 3
    pol = next(d for d in docs if d.doc_id == "insider_trading_policy")
    assert pol.doc_type == "policy"
    assert pol.source == "internal"
    assert "MNPI" in pol.text


def test_doc_type_inferred_from_filename():
    docs = load_internal_docs(Path("data/internal"))
    types = {d.doc_id: d.doc_type for d in docs}
    assert types["data_retention_sop"] == "sop"
    assert types["vendor_contract"] == "contract"
