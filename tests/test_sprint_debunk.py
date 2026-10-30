"""A4: the debunk card leads with the strongest finding and never opens with the claim / raw post title."""
import copy
import json
from pathlib import Path

import pytest

from floodproof import debunk
from floodproof.report import build_report, validate_report
from test_report import proven, CLUES_OK

DEMO = Path(__file__).resolve().parents[1] / "demo"
TITLE = "Massive flood in Rasuwa Nepal today shocking video"


def mk(claim_date="2026-08-26", matches=(), ip=None, title=None, signals_ok=False):
    visual = dict(status="FOUND" if matches else "NOT_FOUND", proven_matches=list(matches), input_publication=ip,
                  earliest_proven_copy=min(matches, key=lambda m: m["upload_date"]) if matches else None)
    claim = dict(text=TITLE, event_date=claim_date, place="Rasuwa")
    inp = dict(source="https://youtu.be/in", url="https://youtu.be/in", title=title or TITLE, description=None, n_keyframes=4)
    r = build_report(input_meta=inp, claim=claim, memory=dict(status="MISS"), visual=visual, clues=CLUES_OK, physical=dict(status="NOT_RUN"))
    if signals_ok:
        r["signals"] = dict(agree=[dict(key="date", label="Date named in the clip", detail="narration says 26 August", field="clues.audio.spoken_dates"),
                                   dict(key="place", label="Place named in the clip", detail="Rasuwa", field="clues.audio.spoken_place_names"),
                                   dict(key="broadcaster", label="Broadcaster logo or credit", detail="NHK", field="clues.visible_text")],
                            against=[], count=3, needed=3, qualifies=True, note="x")
    return r


def ip(date):
    return dict(url="https://youtu.be/in", upload_date=date, date_source="yt-dlp metadata", retrieved_at="t")


def first_sentence(r, lang="en"):
    return debunk.template_debunk(r, [lang])["field_refs"][lang][0]["sentence"]


CASES = {
    "copy": lambda: mk(matches=[proven(date="2024-01-01")], ip=ip("2024-09-28")),
    "ai_label": lambda: mk(title="AI generated flood simulation in Nepal"),
    "input_before": lambda: mk(ip=ip("2024-09-28")),
    "signals": lambda: mk(signals_ok=True),
    "outcome": lambda: mk(),
}


@pytest.mark.parametrize("kind", list(CASES))
def test_lead_kind_and_template_lead_sentence(kind):
    r = CASES[kind]()
    validate_report(r)
    assert debunk.lead_kind(r) == kind
    for lg in ("en", "ne", "hi"):
        f = first_sentence(r, lg)
        assert TITLE.lower() not in f.lower() and not f.lower().startswith(("claim", "the claim", "दाबी", "दावा")), (kind, lg, f)
    en = first_sentence(r)
    if kind == "copy":
        assert "2024-01-01" in en and "Earliest copy found" in en
    elif kind == "ai_label":
        assert "AI-generated" in en
    elif kind == "input_before":
        assert en == "This video was posted on 2024-09-28, before the claimed event on 2026-08-26."
        assert "earlier copy" not in en.lower()
    elif kind == "signals":
        assert "signals agree" in en and "does not prove" in en
    else:
        assert "No earlier copy" in en


@pytest.mark.parametrize("kind", list(CASES))
def test_templates_pass_the_grounding_and_lead_check(kind):
    r = CASES[kind]()
    t = debunk.template_debunk(r, ["en", "ne", "hi"])
    for lg, refs in t["field_refs"].items():
        assert debunk.check_grounded(r, [{"text": x["sentence"], "ref": x["ref"]} for x in refs], lg) == [], (kind, lg)
    # the claim, when shown at all, comes after the lead
    assert t["field_refs"]["en"][0]["ref"] != "claim.text"


def test_input_before_card_never_calls_the_post_date_an_earlier_copy():
    r = CASES["input_before"]()
    txt = debunk.template_debunk(r)["texts"]["en"]
    assert "not an earlier copy found by the matcher" in txt
    assert "Earliest copy found" not in txt
    bad = [{"text": "Earlier copy found: matched on 2024-09-28, before the claimed event on 2026-08-26.", "ref": "input_publication_before_claim.input_date"}]
    assert any("must not be described as an earlier copy" in p for p in debunk.check_grounded(r, bad))


def test_model_card_that_opens_with_the_claim_is_rejected_and_falls_back():
    r = CASES["copy"]()
    ep = r["visual_matching"]["earliest_proven_copy"]
    sents = [{"text": f"The claim is \"{TITLE}\".", "ref": "claim.text"},
             {"text": f"Earliest copy found: published on {ep['upload_date']} ({ep['url']}).", "ref": "visual_matching.earliest_proven_copy.upload_date"}]
    probs = debunk.check_grounded(r, sents)
    assert any("opens with" in p for p in probs)
    d = debunk.generate_debunk(r, chat=lambda m, rr, langs: ({"en": sents, "ne": sents}, 0.0))
    assert d["grounded"] is False and d["model"].startswith("template")
    assert not d["texts"]["en"].lower().startswith("the claim") and TITLE not in d["texts"]["en"][:60]


def test_model_card_must_lead_with_the_ai_label_when_present():
    r = CASES["ai_label"]()
    s = [{"text": "Several sources were consulted for this video.", "ref": "final_verdict.label"}]
    assert any("AI / animation label" in p for p in debunk.check_grounded(r, s))
    ok = [{"text": "The uploader labels this video as AI-generated, so it is not footage of the claimed event.", "ref": "disclosure.type"}]
    assert debunk.check_grounded(r, ok) == []


def test_prompt_demands_the_lead():
    r = CASES["input_before"]()
    p = debunk._prompt(r, ["en", "ne"])
    assert "LEAD KIND: input_before" in p and "Do not start with the claim" in p
    assert "NEVER an 'earlier copy'" in debunk.SYSTEM and "never start with 'The claim is'" in debunk.SYSTEM
    assert "input_publication_before_claim.input_date" in p


def test_facts_never_present_the_input_post_as_an_earliest_copy():
    r = CASES["input_before"]()
    F = debunk.facts(r)
    assert not any(k.startswith("visual_matching.earliest_proven_copy") for k in F)
    assert F["input_publication_before_claim.input_date"] == "2024-09-28"
