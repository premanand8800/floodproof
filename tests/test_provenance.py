"""Independent date re-fetch: caching, fallback order, graceful network failure, enrich_dates semantics."""
from floodproof import provenance, sources


def _iso(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "_FILE", tmp_path / "refetch.json")


def boom(url):
    raise ConnectionError("offline")


def test_fallback_order_and_cache(monkeypatch, tmp_path):
    _iso(monkeypatch, tmp_path)
    calls = []
    methods = (("yt-dlp live metadata", lambda u: calls.append("y") or None), ("page meta", boom),
               ("wayback_cdx_earliest_snapshot", lambda u: calls.append("w") or "2026-08-20"))
    r = provenance.refetch_date("https://x/y", methods=methods)
    assert r["date"] == "2026-08-20" and r["source"] == "wayback_cdx_earliest_snapshot" and calls == ["y", "w"]
    again = provenance.refetch_date("https://x/y", methods=(("none", boom),))
    assert again["cached"] and again["date"] == "2026-08-20"


def test_network_failure_degrades_without_raising_and_is_cached(monkeypatch, tmp_path):
    _iso(monkeypatch, tmp_path)
    r = provenance.refetch_date("https://x/z", methods=(("a", boom), ("b", boom)))
    assert r["date"] is None and r["status"] == "failed" and len(r["errors"]) == 2
    assert provenance.refetch_date("https://x/z", methods=(("a", lambda u: "2020-01-01"),))["cached"]   # failure cached within TTL


def cand(md="2026-08-21"):
    return sources.Candidate("https://ig/reel/1", adapter="memory", status="proven", upload_date=md, date_source="memory entry (inherited)",
                             proven=True, stored_date=md)


def test_enrich_replaces_inherited_date_and_records_conflict():
    c = cand()
    sources.enrich_dates([c], refetch=lambda u: dict(date="2026-08-22", source="yt-dlp live metadata", status="ok", errors=[]))
    assert c.date_independent and c.upload_date == "2026-08-22" and c.stored_date == "2026-08-21"
    assert c.date_alternatives == [dict(date="2026-08-21", source="memory entry (inherited)")]


def test_enrich_failure_keeps_inherited_date_unverified():
    c = cand()
    sources.enrich_dates([c], refetch=lambda u: dict(date=None, status="failed", errors=["offline"]))
    assert not c.date_independent and c.upload_date == "2026-08-21" and "not independently verified" in c.date_note
    sources.enrich_dates([c], refetch=boom)
    assert not c.date_independent
