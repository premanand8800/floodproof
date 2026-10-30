import pytest

from floodproof import physical as P


def make_router(*, ems=None, discharge=(10.0, 10.0), rain=5.0, elev=1300.0, geo=None, ems_error=False):
    """discharge=(baseline, peak) for the centre cell."""
    base, peak = discharge

    def router(url, params):
        if url == P.NOMINATIM:
            return geo if geo is not None else [{"lat": "27.7", "lon": "85.3", "display_name": "Kathmandu",
                                                  "category": "place", "type": "city", "importance": 0.7}]
        if url == P.EMS_LIST:
            if ems_error:
                raise RuntimeError("boom")
            return {"results": ems or [], "next": None}
        if url == P.EMS_DETAIL:
            code = params["code"]
            return {"results": [{"eventTime": "2024-09-28T00:00:00", "reason": "flood " + code}]}
        if url == P.FLOOD_API:
            n = len(params["latitude"].split(","))
            res = []
            for i in range(n):
                times = ["2024-08-%02d" % d for d in range(26, 32)] + ["2024-09-%02d" % d for d in range(1, 26)]
                # window 2024-09-25..2024-10-01 given date 09-28 +-3
                times = ["2024-08-26", "2024-08-27", "2024-09-24", "2024-09-25", "2024-09-26",
                         "2024-09-27", "2024-09-28", "2024-09-29", "2024-09-30", "2024-10-01"]
                vals = [base] * 3 + [base, base, base, peak if i == 4 else base, base, base, base]
                res.append({"latitude": 27.7, "longitude": 85.3, "daily": {"time": times, "river_discharge": vals}})
            return res
        if url == P.ARCHIVE_API:
            times = ["2024-09-25", "2024-09-28", "2024-10-01"]
            return {"daily": {"time": times, "precipitation_sum": [0, rain, 0]}}
        if url == P.ELEV_API:
            return {"elevation": [elev]}
        raise AssertionError(url)

    return router


@pytest.fixture(autouse=True)
def patch(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(P.time, "sleep", lambda s: None)


def run(monkeypatch, router, **kw):
    monkeypatch.setattr(P, "_http_json", router)
    return P.check(place="Kathmandu", date="2024-09-28", **kw)


EMS_NEAR = [{"code": "EMSR1", "name": "Flood near", "category": {"slug": "flood"},
             "centroid": "POINT (85.4 27.8)", "activationTime": "2024-09-29T10:00:00",
             "countries": [{"short_name": "Nepal"}]}]
EMS_FAR = [dict(EMS_NEAR[0], centroid="POINT (100.0 10.0)")]
EMS_QUAKE = [dict(EMS_NEAR[0], category={"slug": "earthquake"})]


def test_supports_via_activation(monkeypatch):
    r = run(monkeypatch, make_router(ems=EMS_NEAR))
    assert r["verdict"] == "SUPPORTS"
    assert r["official_event"]["status"] == "CONFIRMED_EVENT"
    assert r["official_event"]["activations"][0]["code"] == "EMSR1"


def test_supports_via_discharge(monkeypatch):
    r = run(monkeypatch, make_router(discharge=(10.0, 112.0)))
    assert r["verdict"] == "SUPPORTS"
    assert r["official_event"]["status"] == "NO_RECORD"
    assert r["hydrology"]["river_discharge"]["ratio"] == 11.2
    assert r["hydrology"]["status"] == "ANOMALOUS"


def test_supports_via_rain(monkeypatch):
    r = run(monkeypatch, make_router(rain=150))
    assert r["verdict"] == "SUPPORTS"


def test_contradicts_lowland(monkeypatch):
    r = run(monkeypatch, make_router(ems=EMS_FAR + EMS_QUAKE))
    assert r["verdict"] == "CONTRADICTS"
    assert any("weak" in l for l in r["limits"])
    assert r["hydrology"]["status"] == "NORMAL"


def test_no_evidence_high_altitude(monkeypatch):
    r = run(monkeypatch, make_router(elev=3200))
    assert r["verdict"] == "NO_EVIDENCE"


def test_no_evidence_moderate_signal(monkeypatch):
    r = run(monkeypatch, make_router(discharge=(10.0, 20.0)))
    assert r["verdict"] == "NO_EVIDENCE"


def test_low_confidence_unknown(monkeypatch):
    geo = [{"lat": "1", "lon": "1", "display_name": "shop", "category": "shop", "type": "x", "importance": 0.1}]
    r = run(monkeypatch, make_router(geo=geo))
    assert r["location"]["confidence"] == "low"
    assert r["verdict"] == "UNKNOWN"


def test_geocode_failure_unknown(monkeypatch):
    r = run(monkeypatch, make_router(geo=[]) )
    assert r["verdict"] == "UNKNOWN"


def test_ems_error_unknown(monkeypatch):
    r = run(monkeypatch, make_router(ems_error=True))
    assert r["official_event"]["status"] == "ERROR"
    assert r["verdict"] == "UNKNOWN"


def test_given_coords_and_limits(monkeypatch):
    monkeypatch.setattr(P, "_http_json", make_router(discharge=(10, 50)))
    r = P.check(lat=27.7, lon=85.3, date="2024-09-28")
    assert r["location"]["source"] == "given"
    assert r["verdict"] == "SUPPORTS"
    assert P.LIMIT_TEXT in r["limits"]


def test_geocode_prefers_natural(monkeypatch):
    geo = [{"lat": "1", "lon": "1", "display_name": "Bagmati Hotel", "category": "tourism", "type": "hotel", "importance": 0.5},
           {"lat": "2", "lon": "2", "display_name": "Bagmati River", "category": "waterway", "type": "river", "importance": 0.45}]
    monkeypatch.setattr(P, "_http_json", lambda u, p: geo)
    loc = P.geocode("Bagmati")
    assert loc["lat"] == 2.0 and loc["alternatives"][0]["lat"] == 1.0


@pytest.mark.parametrize("et", ["cyclone", "storm_surge", "flash_flood", "glof", "landslide", "unknown", "bogus"])
def test_non_flood_never_contradicts(monkeypatch, et):
    monkeypatch.setattr(P, "_http_json", make_router())
    r = P.check(place="Kathmandu", date="2024-09-28", event_type=et)
    assert r["verdict"] == "NO_EVIDENCE"
    assert any("never CONTRADICTS" in l for l in r["limits"])


def test_non_flood_still_supports(monkeypatch):
    monkeypatch.setattr(P, "_http_json", make_router(ems=EMS_NEAR))
    assert P.check(place="K", date="2024-09-28", event_type="glof")["verdict"] == "SUPPORTS"


def test_geocode_rejects_other_district(monkeypatch):
    calls = []

    def fake(url, params):
        calls.append(params)
        if params["q"] == "Bagmati River":
            return [{"lat": "26", "lon": "85", "display_name": "Bagmati River, Bihar, India",
                     "category": "waterway", "type": "river", "importance": 0.5}]
        return []
    monkeypatch.setattr(P, "_http_json", fake)
    loc = P.geocode("Bagmati River, Kathmandu, Nepal", "np")
    assert loc["lat"] is None
    assert all(c.get("countrycodes") == "np" for c in calls)


def test_geocode_accepts_matching_district(monkeypatch):
    def fake(url, params):
        if params["q"] == "Bagmati River, Kathmandu":
            return [{"lat": "27.7", "lon": "85.3", "display_name": "Bagmati River, Kathmandu, Nepal",
                     "category": "waterway", "type": "river", "importance": 0.5}]
        return []
    monkeypatch.setattr(P, "_http_json", fake)
    assert P.geocode("Bagmati River, Kathmandu, Nepal", "np")["lat"] == 27.7


def test_ems_list_cached(monkeypatch):
    n = {"c": 0}
    base = make_router(ems=EMS_NEAR)

    def counting(url, params):
        n["c"] += url == P.EMS_LIST
        return base(url, params)
    monkeypatch.setattr(P, "_http_json", counting)
    P.check(place="Kathmandu", date="2024-09-28")
    P.check(place="Kathmandu", date="2024-09-28")
    assert n["c"] == 1


def test_country_level_location_never_supports(monkeypatch):
    geo = [{"lat": "28.4", "lon": "84.0", "display_name": "Nepal", "category": "boundary", "type": "administrative",
            "addresstype": "country", "place_rank": 4, "importance": 0.9}]
    r = run(monkeypatch, make_router(geo=geo, ems=EMS_NEAR))
    assert r["location"]["country_level"] and r["location"]["confidence"] == "low"
    assert r["verdict"] == "UNKNOWN" and "whole country" in " ".join(r["reasons"])


# ---------------------------------------------------------------- ReliefWeb + GDELT (mocked)
def _rw_payload(types=("Flood",), date="2024-09-28T00:00:00+00:00"):
    return {"data": [{"id": "1", "fields": {"name": "Nepal: Floods - Sep 2024", "date": {"event": date},
                                            "type": [{"name": t} for t in types], "status": "ongoing",
                                            "url": "https://reliefweb.int/disaster/fl-2024-000001-npl"}}]}


def test_reliefweb_match_supports(monkeypatch):
    monkeypatch.setattr(P, "_http_post_json", lambda url, body: _rw_payload())
    monkeypatch.setattr(P, "_http_json", make_router())
    r = P.check(place="Kathmandu", date="2024-09-28", country="NP")
    assert r["official_event"]["reliefweb"]["status"] == "MATCH"
    assert r["verdict"] == "SUPPORTS"
    assert any("ReliefWeb" in x and "reliefweb.int" in x for x in r["reasons"])


def test_reliefweb_wrong_type_no_match(monkeypatch):
    monkeypatch.setattr(P, "_http_post_json", lambda url, body: _rw_payload(types=("Earthquake",)))
    out = P.reliefweb("NP", P._date(2024, 9, 28), 7, "flood")
    assert out["status"] == "NO_MATCH" and out["disasters"] == []


def test_reliefweb_degrades(monkeypatch):
    def boom(url, body):
        raise RuntimeError("403 Forbidden")
    monkeypatch.setattr(P, "_http_post_json", boom)
    out = P.reliefweb("NP", P._date(2024, 9, 28))
    assert out["status"] == "not_configured" and "appname" in out["reason"]
    assert P.reliefweb(None, P._date(2024, 9, 28))["status"] == "SKIPPED"


def test_reliefweb_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(P, "_http_post_json", lambda url, body: calls.append(1) or _rw_payload())
    P.reliefweb("NP", P._date(2024, 9, 28))
    P.reliefweb("NP", P._date(2024, 9, 28))
    assert len(calls) == 1


def test_gdelt_news_coverage_context_only(monkeypatch):
    arts = [{"url": f"https://n.example/{i}", "title": "t", "domain": "n.example", "seendate": "20240928T000000Z"} for i in range(7)]
    monkeypatch.setattr(P, "_http_json", lambda url, params: {"articles": arts if params["enddatetime"].startswith("20241001") else arts[:2]})
    n = P.news_coverage("Kathmandu, Nepal", P._date(2024, 9, 28), 3, "flood")
    assert n["status"] == "OK" and n["context_only"] and n["label"] == "news coverage"
    assert n["articles_in_window"] == 7 and n["articles_baseline_30d"] == 2
    assert len(n["sample_links"]) == 5


def test_gdelt_alone_never_supports(monkeypatch):
    monkeypatch.setattr(P, "_http_post_json", lambda url, body: {"data": []})
    router = make_router(elev=3200)

    def both(url, params):
        if url == P.GDELT_API:
            return {"articles": [{"url": "https://n.example/a", "title": "t"}] * 50}
        return router(url, params)
    monkeypatch.setattr(P, "_http_json", both)
    r = P.check(place="Kathmandu", date="2024-09-28", country="NP")
    assert r["news_coverage"]["articles_in_window"] == 50
    assert r["verdict"] == "NO_EVIDENCE"


def test_gdelt_degrades(monkeypatch):
    def boom(url, params):
        raise ValueError("not json")
    monkeypatch.setattr(P, "_http_json", boom)
    assert P.news_coverage("Kathmandu", P._date(2024, 9, 28))["status"] == "UNAVAILABLE"
    assert P.news_coverage(None, P._date(2024, 9, 28))["status"] == "SKIPPED"
