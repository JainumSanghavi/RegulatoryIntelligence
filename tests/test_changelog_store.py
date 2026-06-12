from qdrant_client import QdrantClient

from regintel.store.changelog_store import ChangelogStore
from regintel.types import ChangelogEntry


def test_changelog_entry_fields():
    e = ChangelogEntry(accession_no="0001-24-1", title="ACME 8-K", form_type="8-K",
                       filed_date="2026-05-01", url="http://x", summary="new blackout rule",
                       detected_at="2026-06-12T00:00:00+00:00")
    assert e.accession_no == "0001-24-1"
    assert e.summary == "new blackout rule"
    assert e.url == "http://x"


def _store():
    s = ChangelogStore(client=QdrantClient(":memory:"))
    s.ensure_collection()
    return s


def _entry(acc, detected_at):
    return ChangelogEntry(accession_no=acc, title=f"Filing {acc}", form_type="8-K",
                          filed_date="2026-05-01", url=f"http://x/{acc}",
                          summary=f"summary {acc}", detected_at=detected_at)


def test_record_then_is_seen():
    s = _store()
    assert s.is_seen("acc1") is False
    s.record(_entry("acc1", "2026-06-12T00:00:00+00:00"), vector=[0.1] * 1024)
    assert s.is_seen("acc1") is True
    assert s.is_seen("acc2") is False


def test_record_is_idempotent():
    s = _store()
    e = _entry("acc1", "2026-06-12T00:00:00+00:00")
    s.record(e, vector=[0.1] * 1024)
    s.record(e, vector=[0.1] * 1024)
    assert len(s.list_recent()) == 1


def test_list_recent_sorted_desc_by_detected_at():
    s = _store()
    s.record(_entry("a", "2026-06-10T00:00:00+00:00"), vector=[0.1] * 1024)
    s.record(_entry("b", "2026-06-12T00:00:00+00:00"), vector=[0.2] * 1024)
    s.record(_entry("c", "2026-06-11T00:00:00+00:00"), vector=[0.3] * 1024)
    recent = s.list_recent(limit=2)
    assert [e.accession_no for e in recent] == ["b", "c"]
