"""Inferred claim (URL input, empty claim/date/place) and media_nature layer. No network, no LLM."""
from types import SimpleNamespace
import pytest
from floodproof.infer import infer_claim, parse_explicit_date
from floodproof.investigator import _clean
from floodproof.report import build_report, validate_report
from test_report import proven


class F:
    times = [float(i) for i in range(6)]
    frames = [None] * 6
    def __len__(self): return 6


def clip(title="NEPAL FLOOD'S 26 AUG 2026", desc=None, up="2026-08-28", url="https://youtu.be/x"):
    return SimpleNamespace(title=title, description=desc, upload_date=up, url=url)


CLUES = dict(status="ok", claim_parsed=dict(place="Nepal"), visible_text=[], on_screen_dates=[])


def test_date_parsing():
    assert parse_explicit_date("NEPAL FLOOD'S 26 AUG 2026") == "2026-08-26"
    assert parse_explicit_date("August 3rd, 2024") == "2024-08-03"
    assert parse_explicit_date("12/03/2024") is None and parse_explicit_date("no date") is None


def test_explicit_title_date_inferred():
    c = infer_claim(clip(), None, None, None, CLUES)
    assert c["event_date"] == "2026-08-26" and c["event_date_basis"] == "title"
    assert c["source"] == dict(text="inferred_from_post", event_date="inferred_from_post", place="inferred_from_post")
    assert c["place"] == "Nepal" and "inferred from the post" in c["inferred_note"]


def test_on_screen_date_then_date_posted_fallback():
    cl = dict(CLUES, on_screen_dates=[dict(date="2026-08-20", text="20 Aug 2026", refs=[2])])
    c = infer_claim(clip(title="Nepal flood report"), None, None, None, cl)
    assert (c["event_date"], c["event_date_basis"]) == ("2026-08-20", "on_screen_text")
    c = infer_claim(clip(title="Nepal flood report"), None, None, None, CLUES)
    assert (c["event_date"], c["event_date_basis"]) == ("2026-08-28", "date_posted")


def test_user_values_never_overridden_and_ungrounded_place_dropped():
    c = infer_claim(clip(), "my claim", "2020-01-01", "Pokhara", CLUES)
    assert (c["text"], c["event_date"], c["place"]) == ("my claim", "2020-01-01", "Pokhara")
    assert set(c["source"].values()) == {"user"} and c["inferred_note"] is None
    c = infer_claim(clip(title="Big flood"), None, None, None, dict(CLUES, claim_parsed=dict(place="Trishuli Valley")))
    assert c["place"] is None                      # place not stated by the post: a model guess is not inferred
    assert infer_claim(SimpleNamespace(title="t", description=None, upload_date="2026-01-01", url=None), None, None, None, CLUES)["text"] is None


def test_media_nature_needs_cited_frames():
    plan = dict(media_nature=dict(value="animation_or_cgi", confidence=.9, refs=[1, 4, 99], reasons=[dict(text="cartoon", refs=[1])]))
    mn = _clean(plan, F(), False)["media_nature"]
    assert mn["value"] == "animation_or_cgi" and mn["cited_frames"] == [1, 4] and mn["status"] == "AI_OBSERVATION_HYPOTHESIS"
    assert _clean(dict(media_nature=dict(value="animation_or_cgi", refs=[])), F(), False)["media_nature"]["value"] == "unknown"
    assert _clean(dict(media_nature=dict(value="bogus", refs=[1])), F(), False)["media_nature"]["value"] == "unknown"
    d = _clean(dict(on_screen_dates=[dict(date="2026-08-26", text="x", refs=[1]), dict(date="soon", text="y", refs=[1])]), F(), False)
    assert [x["date"] for x in d["on_screen_dates"]] == ["2026-08-26"]


def _rep(nature, claim, matches, clues_extra=None):
    mn = dict(value=nature, confidence=.8, cited_frames=[1, 3], cited_timestamps_s=[1.0, 3.0], reasons=[], status="AI_OBSERVATION_HYPOTHESIS")
    clues = dict(status="ok", observations_against_claim=[], location_hypotheses=[], media_nature=mn)
    visual = dict(status="FOUND" if matches else "NOT_FOUND", proven_matches=matches,
                  input_publication=dict(url="https://youtu.be/x", upload_date="2026-08-28", date_source="yt-dlp metadata"))
    return build_report(input_meta=dict(source="x", url="https://youtu.be/x"), claim=claim, memory=dict(status="MISS"),
                        visual=visual, clues=clues, physical=dict(status="NOT_RUN"))


def test_compilation_earlier_copy_wording_and_posted_date_never_recycles():
    claim = dict(text="t", event_date="2026-08-28", event_date_basis="date_posted", inferred_note="Claim inferred from the post's title/caption (you did not enter one)",
                 source=dict(text="inferred_from_post", event_date="inferred_from_post", place=None))
    r = validate_report(_rep("news_compilation", claim, [proven(date="2026-08-26")]))
    fv = r["final_verdict"]
    assert fv["label"] == "UNVERIFIED" and "contains a clip published earlier" in fv["summary"]
    assert "AI observation" in fv["summary"] and "do not by themselves mean recycling" in fv["summary"]
    assert r["schema_version"] == "1.5" and r["media_nature"]["value"] == "news_compilation" and not r["contradictions"]


def test_media_nature_never_changes_label():
    claim = dict(text="t", event_date="2026-08-26", event_date_basis="title", source=dict(event_date="inferred_from_post"))
    labels = {n: validate_report(_rep(n, claim, []))["final_verdict"]["label"] for n in ("camera_footage", "animation_or_cgi", "unknown")}
    assert len(set(labels.values())) == 1
    s = validate_report(_rep("animation_or_cgi", claim, []))["final_verdict"]["summary"]
    assert "animated / reenactment" in s and "frames 1,3" in s


def test_validation_rules():
    claim = dict(text="t", event_date="2026-08-26", event_date_basis="title", source=dict(event_date="inferred_from_post"))
    r = _rep("animation_or_cgi", claim, [])
    r["media_nature"]["cited_frames"] = []
    with pytest.raises(AssertionError):
        validate_report(r)
    r = _rep("unknown", claim, []); r["input"] = dict(source="x")
    with pytest.raises(AssertionError):
        validate_report(r)                           # inferred claim needs a URL input


def test_schema_1_1_demo_still_validates():
    r = _rep("unknown", dict(text="t", event_date=None), [])
    r["schema_version"] = "1.1"; r.pop("media_nature")
    validate_report(r)


def test_inferred_date_not_confirmed_by_own_post():
    # claim date inferred from the post; only evidence is the post's own publication 3 days later -> circular, stays UNVERIFIED
    claim = dict(text="t", event_date="2026-08-26", event_date_basis="spoken_in_narration_partial_year",
                 inferred_note="Claim inferred from the post's title/caption (you did not enter one)",
                 source=dict(text="inferred_from_post", event_date="inferred_from_post", place=None))
    r = validate_report(_rep("camera_footage", claim, []))
    assert r["final_verdict"]["label"] == "UNVERIFIED"
