from regintel.types import ChangelogEntry


def test_changelog_entry_fields():
    e = ChangelogEntry(accession_no="0001-24-1", title="ACME 8-K", form_type="8-K",
                       filed_date="2026-05-01", url="http://x", summary="new blackout rule",
                       detected_at="2026-06-12T00:00:00+00:00")
    assert e.accession_no == "0001-24-1"
    assert e.summary == "new blackout rule"
    assert e.url == "http://x"
