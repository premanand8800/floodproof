"""Debunk card: grounded in report.json, verdict-respecting, deterministic fallback; LLM is always mocked."""
import copy
import json
from pathlib import Path

import pytest

from floodproof import debunk
from floodproof.report import validate_report

DEMO = Path(__file__).resolve().parents[1] / "demo"


def load(name):
    r = json.loads((DEMO / name / "report.json").read_text())
    r.pop("debunk", None)
    return r


def good_obj(r):
    ep = r["visual_matching"]["earliest_proven_copy"]
    s = lambda t, ref: {"text": t, "ref": ref}
    one = [s(f"Earliest copy found: published on {ep['upload_date']} ({ep['url']}).", "visual_matching.earliest_proven_copy.upload_date"),
           s(f"The matcher found {ep['match']['consistent_frames']} consistent frames.", "visual_matching.earliest_proven_copy.match.consistent_frames"),
           s("This does not show the event did not happen.", "final_verdict.label")]
    return {"en": one, "ne": [dict(x, text=x["text"] + " (ne)") for x in one]}


def test_good_output_is_grounded():
    r = load("success_recycled")
    d = debunk.generate_debunk(r, chat=lambda m, rr, langs: (good_obj(r), 0.001))
    assert d["grounded"] is True and d["model"] == debunk.MODEL
    assert set(d["texts"]) == {"en", "ne"} and all(x["ref"] for x in d["field_refs"]["en"])
    r["debunk"] = d
    validate_report(r)                      # optional field accepted


def test_rejects_invented_date_and_falls_back_to_template():
    r = load("success_recycled")
    bad = good_obj(r)
    bad["en"][0]["text"] = "Earliest copy found: published on 2026-08-19 (https://www.instagram.com/reel/DcUFQZhvGzS/)."
    assert any("date not in report" in p for p in debunk.check_grounded(r, bad["en"]))
    d = debunk.generate_debunk(r, chat=lambda m, rr, langs: (bad, 0.001))
    assert d["grounded"] is False and d["model"].startswith("template")
    assert "2026-08-21" in d["texts"]["en"] and "2026-08-19" not in d["texts"]["en"] and "2026-08-19" not in d["texts"]["ne"]


def test_rejects_invented_url_number_and_bad_ref():
    r = load("success_recycled")
    ok = good_obj(r)["en"]
    assert debunk.check_grounded(r, ok) == []
    u = copy.deepcopy(ok); u[0]["text"] += " See https://evil.example/x"
    n = copy.deepcopy(ok); n[1]["text"] = "The matcher found 777 consistent frames."
    b = copy.deepcopy(ok); b[2]["ref"] = "visual_matching.nope"
    assert any("URL" in p for p in debunk.check_grounded(r, u))
    assert any("number" in p for p in debunk.check_grounded(r, n))
    assert any("ref does not resolve" in p for p in debunk.check_grounded(r, b))


def test_devanagari_digits_are_checked_too():
    r = load("success_recycled")
    s = [{"text": "यो भिडियो २०२६-०८-१९ मा प्रकाशित भयो।", "ref": "final_verdict.label"}]
    assert any("date not in report" in p for p in debunk.check_grounded(r, s, "ne"))


def test_never_original_or_fake():
    r = load("success_recycled")
    s = [{"text": "This is the original and it is fake.", "ref": "final_verdict.label"}]
    assert sum("banned" in p for p in debunk.check_grounded(r, s)) == 2


@pytest.mark.parametrize("name", ["failed_unverified", "genuine_correct_claim"])
def test_non_recycled_verdicts_never_say_recycled(name):
    r = load(name)
    s = [{"text": "This footage is recycled.", "ref": "final_verdict.label"}]
    assert any("recycling wording" in p for p in debunk.check_grounded(r, s))
    t = debunk.template_debunk(r)
    for lg, txt in t["texts"].items():
        assert "fake" not in txt.lower() and "original" not in txt.lower() and "recycl" not in txt.lower()
    assert "No earlier copy" in t["texts"]["en"] or "No contradiction" in t["texts"]["en"]


def test_recycled_verdict_requires_earliest_date():
    r = load("success_recycled")
    s = [{"text": "The footage was found elsewhere.", "ref": "final_verdict.label"}]
    assert any("earliest copy date missing" in p for p in debunk.check_grounded(r, s))


def test_fallback_model_used_when_primary_errors():
    r = load("success_recycled")
    calls = []

    def chat(model, rr, langs):
        calls.append(model)
        if model == debunk.MODEL:
            raise RuntimeError("boom")
        return good_obj(rr), 0.002
    d = debunk.generate_debunk(r, chat=chat)
    assert d["model"] == debunk.FALLBACK_MODEL and d["grounded"] is True and calls[0] == debunk.MODEL


def test_all_templates_pass_their_own_check_and_cover_all_demos():
    for name in ("success_recycled", "failed_unverified", "genuine_correct_claim", "genuine_wrong_date"):
        r = load(name)
        t = debunk.template_debunk(r, ["en", "ne", "hi"])
        for lg, refs in t["field_refs"].items():
            assert debunk.check_grounded(r, [{"text": x["sentence"], "ref": x["ref"]} for x in refs], lg) == [], (name, lg)


def test_detects_rumour_language():
    r = load("success_recycled")
    assert debunk.detect_langs(r) == ["en", "ne"]
    r["clues"] = {"audio": {"status": "ok", "language": "Hindi", "has_speech": True}}
    assert debunk.detect_langs(r) == ["en", "ne", "hi"]


def test_validate_report_debunk_optional_and_checked():
    r = load("success_recycled")
    validate_report(r)                                            # absent: fine (backward compatible)
    r["debunk"] = {"texts": {"en": "x"}, "model": "m", "grounded": True, "field_refs": {}}
    with pytest.raises(AssertionError):
        validate_report(r)


def test_committed_demo_debunks_are_still_grounded():
    for name in ("success_recycled", "failed_unverified", "genuine_correct_claim", "genuine_wrong_date"):
        r = json.loads((DEMO / name / "report.json").read_text())
        d = r["debunk"]
        validate_report(r)
        for lg, refs in d["field_refs"].items():
            assert debunk.check_grounded(r, [{"text": x["sentence"], "ref": x["ref"]} for x in refs], lg) == []


def test_headline_numbers_from_bench_files():
    from floodproof import web
    h = web.bench_headline()
    assert (h["clips"], h["earlier_copy_proven"], h["false_accusations"]) == (17, 10, 0) and h["cents_per_check"] == 1.5
