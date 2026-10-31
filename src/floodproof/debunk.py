"""Shareable debunk card (English + Nepali + the rumour's language), generated FROM report.json only.

Every sentence cites the report field it comes from. A post-check rejects the model output if any date, URL or number is
not present in the report, if a cited field does not exist, or if the wording breaks the rules ("original", "fake",
or recycling language when the verdict is not RECYCLED_FOOTAGE). On any failure a deterministic template is used and
`grounded` is set to False for the LLM path (the template is grounded by construction: it is filled from report fields).

  report["debunk"] = {"texts": {lang: text}, "field_refs": {lang: [ref, ...]}, "model": str, "grounded": bool, ...}
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from . import llm

MODEL = "meta/muse-spark-open"          # open weights, served by Baseten first (OpenRouter fallback); 8/8 demo cards grounded on 2026-10-31
GLIMMER_MODEL = "meta/muse-glimmer-30b"
FALLBACK_MODEL = "meta/muse-spark-1.3"
BASE_LANGS = ("en", "ne")
LANG_NAMES = {"en": "English", "ne": "Nepali", "hi": "Hindi"}
# Narration language name (as the audio layer reports it) -> code. Unknown names are used as-is, lower-cased.
LANG_CODES = {"english": "en", "nepali": "ne", "hindi": "hi", "urdu": "ur", "bengali": "bn", "bangla": "bn", "tamil": "ta",
              "telugu": "te", "marathi": "mr", "gujarati": "gu", "punjabi": "pa", "arabic": "ar", "spanish": "es", "french": "fr",
              "indonesian": "id", "burmese": "my", "thai": "th", "portuguese": "pt", "russian": "ru", "turkish": "tr"}
BANNED = [r"\boriginal\b", r"\bfake[sd]?\b", r"\bhoax\b", r"नक्कली", r"नकली", r"फर्जी", r"झुटो", r"मौलिक", r"असली", r"\bfalsified\b"]
RECYCLE_WORDS = [r"\brecycl", r"\breus", r"\bre-?upload", r"पुन:?प्रयोग", r"पुरानो भिडियो", r"पुराना वीडियो"]
_DEV = str.maketrans("०१२३४५६७८९", "0123456789")
URL_RE = re.compile(r"https?://[^\s)\]»”\"']+")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUM_RE = re.compile(r"\d+")


# ---------------------------------------------------------------- facts from the report
def _get(r, path):
    cur = r
    for p in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(p)]
        else:
            cur = cur[p]
    return cur


def facts(r: dict) -> dict:
    """{field_path: value} for the report fields a debunk may cite. Paths resolve in report.json."""
    v, fv, cm = r["visual_matching"], r["final_verdict"], r["claim"]
    F = {"final_verdict.label": fv["label"]}
    for k in ("text", "event_date", "place"):
        if cm.get(k):
            F[f"claim.{k}"] = cm[k]
    if posted_only(r) and cm.get("event_date"):
        # the claim "date" is only the day the post was uploaded: never present it as the date of an event
        F["claim.event_date"] = f"{cm['event_date']} (POST DATE ONLY: the video was posted on this date; no event date was stated)"
    ep = v.get("earliest_proven_copy")
    base = "visual_matching.earliest_proven_copy"
    if ep:
        for k in ("url", "upload_date", "uploader"):
            if ep.get(k):
                F[f"{base}.{k}"] = ep[k]
        if ep.get("date_source"):
            F[f"{base}.date_source"] = ep["date_source"]
        m = ep.get("match") or {}
        for k in ("consistent_frames", "max_inliers"):
            if m.get(k) is not None:
                F[f"{base}.match.{k}"] = m[k]
        if "date_independent" in ep:
            F[f"{base}.date_independent"] = ep["date_independent"]
    F["visual_matching.status"] = v["status"]
    # the other leading findings (A4): the input video's own publication before the claim (NOT an earlier copy), the uploader's own AI label, agreeing signals
    ibc = r.get("input_publication_before_claim")
    if ibc:
        for k in ("input_date", "claim_date", "url"):
            if ibc.get(k):
                F[f"input_publication_before_claim.{k}"] = ibc[k]
    dsc = r.get("disclosure") or {}
    if dsc.get("type", "none") != "none":
        F["disclosure.type"] = dsc["type"]
        ev0 = (dsc.get("evidence") or [{}])[0]
        if ev0.get("text"):
            F["disclosure.evidence.0.text"] = ev0["text"]
    sg = r.get("signals") or {}
    if sg.get("qualifies"):
        F["signals.count"] = sg["count"]
        for i, a in enumerate((sg.get("agree") or [])[:3]):
            F[f"signals.agree.{i}.label"] = a["label"]
    mn = r.get("media_nature") or {}
    if mn.get("value") == "mixed" and mn.get("summary"):
        F["media_nature.summary"] = mn["summary"]
    return {k: val for k, val in F.items() if val is not None}


def has_earlier_copy(r: dict) -> bool:
    """A matcher-proven, independently dated copy that predates the claim by the time-zone margin (the input's own post never counts)."""
    from .report import proven_before, claim_date_for_logic
    return bool(proven_before((r["visual_matching"].get("proven_matches") or []), claim_date_for_logic(r["claim"]), independent_only=True))


LEAD_KINDS = ("copy", "ai_label", "input_before", "signals", "outcome")


def lead_kind(r: dict) -> str:
    """The strongest finding, which the card must lead with: a proven earlier copy > the uploader's own AI label > the input video posted
    before the claimed event > agreeing signals > otherwise a plain statement of what was and was not found. Never the raw claim/post title."""
    F = facts(r)
    lab = F["final_verdict.label"]
    if lab == "RECYCLED_FOOTAGE" and has_earlier_copy(r) and "visual_matching.earliest_proven_copy.upload_date" in F:
        return "copy"
    if "disclosure.type" in F:
        return "ai_label"
    if lab == "RECYCLED_FOOTAGE" and "input_publication_before_claim.input_date" in F:
        return "input_before"
    if "signals.count" in F:
        return "signals"
    return "outcome"


LEAD_INSTRUCTION = {
    "copy": "State the earliest proven copy first: the date it was published, where, and that this is before the claimed event date.",
    "ai_label": "State first that the uploader's own post labels this video AI-generated / an animation / a reenactment (see disclosure.type), so it is not documentary footage of the claimed event.",
    "input_before": "State first, using exactly this shape: 'This video was posted on <input_date>, before the claimed event on <claim_date>.' Do NOT call this an earlier copy or a match: it is the video's own post date.",
    "signals": "State first that several independent signals agree with the claim (name them from signals.agree.*.label) and that this does not prove the footage is genuine.",
    "outcome": "State first, plainly, what was and was not found (for example that no earlier copy was found, and what that does and does not mean).",
}


def posted_only(r: dict) -> bool:
    """True when claim.event_date is just the post (upload) date, not a stated event date."""
    return (r.get("claim") or {}).get("event_date_basis") == "date_posted"


def detect_langs(r: dict) -> list[str]:
    """Always en + ne; plus the rumour's language when the narration analysis detected another one."""
    langs = list(BASE_LANGS)
    name = str(((r.get("clues") or {}).get("audio") or {}).get("language") or "").strip().lower()
    if name and name not in ("unknown", "none", "n/a") and (r.get("clues") or {}).get("audio", {}).get("has_speech", True) is not False:
        code = LANG_CODES.get(name, re.sub(r"[^a-z]", "", name)[:12])
        if code and code not in langs:
            langs.append(code)
    return langs


def _host(url):
    h = urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


# ---------------------------------------------------------------- deterministic templates
def _template(r: dict, lang: str):
    """Returns [(sentence, ref)] or None when there is no template for this language."""
    F = facts(r)
    lab = F["final_verdict.label"]
    ep = next((k[:-4] for k in F if k.endswith(".url") and k.startswith("visual_matching.")), None)
    claim = F.get("claim.text")
    ev = (r["claim"].get("event_date") if not posted_only(r) else None) or None
    ev_posted = r["claim"].get("event_date") if posted_only(r) else None
    S = []
    T = {
        "en": dict(
            claim=lambda c: f"Claim checked: \"{c}\".",
            ai=lambda t: {"ai_generated": "The uploader's own post labels this video as AI-generated, so it is not documentary footage of the claimed event.",
                          "animation": "The uploader's own post describes this video as an animation, so it is not documentary footage of the claimed event.",
                          "reenactment": "The uploader's own post describes this video as a reenactment or illustration, so it is not documentary footage of the claimed event."}[t],
            inb=lambda d1, d2: f"This video was posted on {d1}, before the claimed event on {d2}.",
            inb2="This is the video's own post date, not an earlier copy found by the matcher. It does not show that the event did not happen.",
            sig=lambda n, ls: f"{n} signals agree with the claim ({', '.join(ls)}), but this does not prove the footage is genuine.",
            nat=lambda sm: f"{sm}. This is an AI observation from sampled frames, not proof.",
            posted=lambda d: f"The video was posted on {d}; no event date was stated.",
            copy=lambda d, u, who, ev, ind: (f"Earliest copy found: this footage was published on {d}" + (f" by {who}" if who else "") + f" ({u})"
                                              + (f", before the claimed event date {ev}" if ev and d < ev else "") + "."
                                              + ("" if ind else " Its date is not independently verified.")),
            match=lambda n, i: f"The frame matcher found {n} consistent frames and up to {i} matching keypoints." if n and i else None,
            r1="This shows the footage was published earlier. It does not show that the event did not happen.",
            r2="This is the earliest copy found, not necessarily the first upload.",
            u1="No earlier copy of this footage was found by the tools used. This does not mean the video is genuine or wrong; the available evidence could not settle it.",
            u2=lambda d, u: f"A matching copy dated {d} was found ({u}), but it does not establish that the claim is wrong.",
            u3="Search is not exhaustive: not finding an earlier copy is a normal outcome.",
            c1="No contradiction with the claim was found in the evidence available. This does not confirm that the video is authentic.",
            c2=lambda d, u: f"A matching copy dated {d} was found ({u}); it is not earlier than the claimed event.",
            c3="Search is not exhaustive."),
        "ne": dict(
            claim=lambda c: f"जाँच गरिएको दाबी: “{c}”।",
            ai=lambda t: {"ai_generated": "अपलोडरको आफ्नै पोस्टले यो भिडियोलाई AI-निर्मित भनेको छ, त्यसैले यो दाबी गरिएको घटनाको वास्तविक फुटेज होइन।",
                          "animation": "अपलोडरको आफ्नै पोस्टले यो भिडियोलाई एनिमेसन भनेको छ, त्यसैले यो दाबी गरिएको घटनाको वास्तविक फुटेज होइन।",
                          "reenactment": "अपलोडरको आफ्नै पोस्टले यो भिडियोलाई पुनःनिर्माण वा चित्रण भनेको छ, त्यसैले यो दाबी गरिएको घटनाको वास्तविक फुटेज होइन।"}[t],
            inb=lambda d1, d2: f"यो भिडियो {d1} मा पोस्ट गरिएको थियो, जुन दाबी गरिएको घटना ({d2}) भन्दा अघि हो।",
            inb2="यो भिडियोकै आफ्नै पोस्ट मिति हो, म्याचरले भेटेको पुरानो प्रति होइन। यसले घटना नभएको भन्ने देखाउँदैन।",
            sig=lambda n, ls: f"{n} संकेतहरू दाबीसँग मेल खान्छन् ({', '.join(ls)}), तर यसले भिडियो साँचो हो भन्ने प्रमाणित गर्दैन।",
            nat=lambda sm: "AI अवलोकन (प्रमाण होइन): भिडियोमा AI-निर्मित जस्ता देखिने दृश्य र वास्तविक क्यामेरा फुटेज मिसिएका हुन सक्छन्।",
            posted=lambda d: f"भिडियो {d} मा पोस्ट गरिएको थियो; घटनाको मिति उल्लेख गरिएको थिएन।",
            copy=lambda d, u, who, ev, ind: (f"भेटिएको सबैभन्दा पुरानो प्रति: यो भिडियो {d} मा" + (f" {who} द्वारा" if who else "") + f" ({u}) प्रकाशित भएको थियो"
                                             + (f", दाबी गरिएको घटनाको मिति {ev} भन्दा अघि" if ev and d < ev else "") + "।"
                                             + ("" if ind else " यसको मिति स्वतन्त्र रूपमा पुष्टि भएको छैन।")),
            match=lambda n, i: f"फ्रेम मिलानमा {n} क्रमिक फ्रेम र {i} सम्म मिल्ने कीपोइन्ट भेटिए।" if n and i else None,
            r1="यसले भिडियो पहिले नै प्रकाशित भएको देखाउँछ; घटना नभएको भन्ने यसले देखाउँदैन।",
            r2="यो भेटिएको सबैभन्दा पुरानो प्रति मात्र हो, पहिलो अपलोड नै हो भन्ने होइन।",
            u1="प्रयोग गरिएका उपकरणले यो भिडियोको पुरानो प्रति भेटेनन्। यसको अर्थ भिडियो साँचो वा गलत भन्ने होइन; उपलब्ध प्रमाणले निर्णय गर्न सकेन।",
            u2=lambda d, u: f"{d} मितिको मिल्दो प्रति भेटियो ({u}), तर यसले दाबी गलत हो भन्ने प्रमाणित गर्दैन।",
            u3="खोजी पूर्ण छैन; पुरानो प्रति नभेट्नु सामान्य नतिजा हो।",
            c1="उपलब्ध प्रमाणमा दाबीसँग बाझिने कुरा भेटिएन। यसले भिडियो प्रामाणिक हो भन्ने पुष्टि गर्दैन।",
            c2=lambda d, u: f"{d} मितिको मिल्दो प्रति भेटियो ({u}); यो दाबी गरिएको घटनाभन्दा अघिको होइन।",
            c3="खोजी पूर्ण छैन।"),
        "hi": dict(
            claim=lambda c: f"जाँचा गया दावा: “{c}”।",
            ai=lambda t: {"ai_generated": "अपलोडर की अपनी पोस्ट इस वीडियो को AI से बना हुआ बताती है, इसलिए यह दावा की गई घटना का वास्तविक फुटेज नहीं है।",
                          "animation": "अपलोडर की अपनी पोस्ट इस वीडियो को एनिमेशन बताती है, इसलिए यह दावा की गई घटना का वास्तविक फुटेज नहीं है।",
                          "reenactment": "अपलोडर की अपनी पोस्ट इस वीडियो को पुनर्निर्माण या चित्रण बताती है, इसलिए यह दावा की गई घटना का वास्तविक फुटेज नहीं है।"}[t],
            inb=lambda d1, d2: f"यह वीडियो {d1} को पोस्ट किया गया था, जो दावा की गई घटना ({d2}) से पहले है।",
            inb2="यह वीडियो की अपनी पोस्ट तारीख है, मैचर को मिली कोई पुरानी प्रति नहीं। इससे यह साबित नहीं होता कि घटना नहीं हुई।",
            sig=lambda n, ls: f"{n} संकेत दावे से मेल खाते हैं ({', '.join(ls)}), लेकिन इससे यह साबित नहीं होता कि फुटेज वास्तविक है।",
            nat=lambda sm: "AI अवलोकन (सबूत नहीं): वीडियो में AI से बने जैसे दिखने वाले शॉट और असली कैमरा फुटेज मिले हो सकते हैं।",
            posted=lambda d: f"वीडियो {d} को पोस्ट किया गया था; घटना की कोई तारीख नहीं बताई गई थी।",
            copy=lambda d, u, who, ev, ind: (f"मिली सबसे पुरानी प्रति: यह वीडियो {d} को" + (f" {who} द्वारा" if who else "") + f" ({u}) प्रकाशित हुआ था"
                                             + (f", दावा किए गए घटना की तारीख {ev} से पहले" if ev and d < ev else "") + "।"
                                             + ("" if ind else " इसकी तारीख स्वतंत्र रूप से सत्यापित नहीं है।")),
            match=lambda n, i: f"फ्रेम मिलान में {n} लगातार फ्रेम और {i} तक मेल खाते कीपॉइंट मिले।" if n and i else None,
            r1="इससे पता चलता है कि वीडियो पहले प्रकाशित हो चुका था; इससे यह साबित नहीं होता कि घटना नहीं हुई।",
            r2="यह मिली सबसे पुरानी प्रति है, ज़रूरी नहीं कि पहला अपलोड हो।",
            u1="इस्तेमाल किए गए टूल को इस वीडियो की कोई पुरानी प्रति नहीं मिली। इसका मतलब यह नहीं कि वीडियो सही या गलत है; उपलब्ध सबूत से फ़ैसला नहीं हो सका।",
            u2=lambda d, u: f"{d} की एक मिलती-जुलती प्रति मिली ({u}), लेकिन इससे दावा गलत साबित नहीं होता।",
            u3="खोज पूरी नहीं है; पुरानी प्रति न मिलना सामान्य नतीजा है।",
            c1="उपलब्ध सबूतों में दावे से टकराने वाली कोई बात नहीं मिली। इससे वीडियो के प्रामाणिक होने की पुष्टि नहीं होती।",
            c2=lambda d, u: f"{d} की एक मिलती-जुलती प्रति मिली ({u}); यह दावा किए गए घटना से पहले की नहीं है।",
            c3="खोज पूरी नहीं है."),
    }
    t = T.get(lang)
    if t is None:
        return None
    d, u = (F.get(f"{ep}.upload_date"), F.get(f"{ep}.url")) if ep else (None, None)
    kind = lead_kind(r)
    # 1. the lead: the strongest finding. The claim text never comes first.
    if kind == "ai_label":
        S.append((t["ai"](F["disclosure.type"]), "disclosure.type"))
    elif kind == "input_before":
        S.append((t["inb"](F["input_publication_before_claim.input_date"], F["input_publication_before_claim.claim_date"]), "input_publication_before_claim.input_date"))
        S.append((t["inb2"], "input_publication_before_claim.claim_date"))
    elif kind == "signals":
        S.append((t["sig"](F["signals.count"], [F[k] for k in sorted(F) if k.startswith("signals.agree.") and k.endswith(".label")]), "signals.count"))
    # 2. the verdict-specific sentences
    V = []
    if lab == "RECYCLED_FOOTAGE" and ep and d and u and has_earlier_copy(r):
        V.append((t["copy"](d, u, F.get(f"{ep}.uploader"), ev, F.get(f"{ep}.date_independent", True)), f"{ep}.upload_date"))
        mt = t["match"](F.get(f"{ep}.match.consistent_frames"), F.get(f"{ep}.match.max_inliers"))
        if mt:
            V.append((mt, f"{ep}.match.consistent_frames"))
        V += [(t["r1"], "final_verdict.label"), (t["r2"], "visual_matching.status")]
    elif lab == "RECYCLED_FOOTAGE" and "input_publication_before_claim.input_date" in F:
        if kind != "input_before":           # the lead is something else (the uploader's AI label): still state the post date
            V.append((t["inb"](F["input_publication_before_claim.input_date"], F["input_publication_before_claim.claim_date"]), "input_publication_before_claim.input_date"))
        V.append((t["u3"], "final_verdict.label"))
    elif lab == "CONSISTENT_WITH_CLAIM":
        V.append((t["c1"], "final_verdict.label"))
        if ep and d and u:
            V.append((t["c2"](d, u), f"{ep}.upload_date"))
        V.append((t["c3"], "visual_matching.status"))
    else:                                    # UNVERIFIED (and a RECYCLED label without a citable copy degrades to this)
        if ep and d and u:
            V.append((t["u2"](d, u), f"{ep}.upload_date"))
        else:
            V.append((t["u1"], "visual_matching.status"))
        V.append((t["u3"], "final_verdict.label"))
    # 3. the claim and the posted-date note come AFTER the lead sentence(s), never first
    if kind in ("copy", "outcome"):
        S, V = V[:1], V[1:]
    if claim:
        S.append((t["claim"](claim), "claim.text"))
    if ev_posted:
        S.append((t["posted"](ev_posted), "claim.event_date"))
    S += V
    if "media_nature.summary" in F:
        S.insert(min(len(S), 2), (t["nat"](F["media_nature.summary"]), "media_nature.summary"))
    return S


def template_debunk(r: dict, langs=None) -> dict:
    texts, refs = {}, {}
    for lg in langs or BASE_LANGS:
        S = _template(r, lg) or _template(r, "en")
        texts[lg] = " ".join(s for s, _ in S)
        refs[lg] = [{"sentence": s, "ref": ref} for s, ref in S]
    return dict(texts=texts, field_refs=refs, model="template", grounded=True, source="deterministic_template")


# ---------------------------------------------------------------- grounding post-check
def _norm(s: str) -> str:
    return str(s).translate(_DEV)


def check_grounded(r: dict, sentences: list[dict], lang: str = "en") -> list[str]:
    """Problems found in model sentences [{text, ref}] vs the report; [] means grounded."""
    problems = []
    blob = _norm(json.dumps(r, ensure_ascii=False, default=str))
    nums = {int(x) for x in NUM_RE.findall(blob)}
    F = facts(r)
    lab = r["final_verdict"]["label"]
    post_date = r["claim"].get("event_date") if posted_only(r) else None
    if not sentences:
        return ["no sentences"]
    for s in sentences:
        text, ref = _norm(s.get("text", "")), s.get("ref")
        if not text.strip():
            problems.append("empty sentence"); continue
        try:
            _get(r, ref)
        except Exception:
            problems.append(f"ref does not resolve in report: {ref!r}"); continue
        for u in URL_RE.findall(text):
            u = u.rstrip(".,;")
            if u not in blob:
                problems.append(f"URL not in report: {u}")
        for d in DATE_RE.findall(text):
            if d not in blob:
                problems.append(f"date not in report: {d}")
            elif post_date and d == post_date and not (
                    re.search(r"posted\s+(on\s+)?" + re.escape(d), text, re.I) if lang == "en" else ref == "claim.event_date"):
                problems.append(f"post date {d} presented as an event date (it is only the date posted)")
        stripped = DATE_RE.sub(" ", URL_RE.sub(" ", text))
        for n in NUM_RE.findall(stripped):
            if int(n) not in nums:
                problems.append(f"number not in report: {n}")
        # dd Month yyyy style dates: the year and day must still be report numbers (covered above); months are not facts.
        for pat in BANNED:
            if re.search(pat, s.get("text", ""), re.I):
                problems.append(f"banned wording /{pat}/")
        if lab != "RECYCLED_FOOTAGE":
            for pat in RECYCLE_WORDS:
                if re.search(pat, s.get("text", ""), re.I):
                    problems.append(f"recycling wording with verdict {lab}")
    joined = " ".join(_norm(s.get("text", "")) for s in sentences)
    if lab == "RECYCLED_FOOTAGE":
        d = (F.get("visual_matching.earliest_proven_copy.upload_date") if has_earlier_copy(r) else None) or F.get("input_publication_before_claim.input_date")
        if d and d not in joined:
            problems.append("RECYCLED verdict but earliest copy date missing")
    problems += check_lead(r, sentences, lang)
    return problems


_AI_WORDS = re.compile(r"\bAI\b|artificial|animat|reenact|illustrat|generated|एआई|एनिमेसन|एनिमेशन|AI-|पुनः?निर्माण|चित्रण", re.I)


def check_lead(r: dict, sentences: list[dict], lang: str = "en") -> list[str]:
    """The card must LEAD with the strongest finding and never open with the claim or the raw post title (A4)."""
    if not sentences:
        return []
    first = _norm(sentences[0].get("text", "")).strip()
    probs = []
    claim = str((r.get("claim") or {}).get("text") or "").strip()
    if claim and len(claim) >= 10 and claim.lower()[:40] in first.lower():
        probs.append("card opens with the claim / post title instead of the strongest finding")
    if re.match(r"^\W*(the\s+)?claim\b", first, re.I) or re.match(r"^\W*(दाबी|दावा)", first):
        probs.append("card opens with 'the claim' instead of the strongest finding")
    if lang == "en":
        kind, F = lead_kind(r), facts(r)
        if kind == "copy" and F.get("visual_matching.earliest_proven_copy.upload_date", "?") not in first:
            probs.append("lead must state the earliest proven copy's date")
        elif kind == "input_before":
            i, c = F["input_publication_before_claim.input_date"], F["input_publication_before_claim.claim_date"]
            if i not in first or c not in first:
                probs.append("lead must say the video was posted on <date> before the claimed event on <date>")
            if re.search(r"earlier copy|matched|match", first, re.I):
                probs.append("input post date must not be described as an earlier copy / match")
        elif kind == "ai_label" and not _AI_WORDS.search(first):
            probs.append("lead must state the uploader's AI / animation label")
        elif kind == "signals" and not re.search(r"signal", first, re.I):
            probs.append("lead must state that signals agree")
    return probs


# ---------------------------------------------------------------- LLM generation
SYSTEM = ("You write short, shareable fact-check cards from a structured report. You may use ONLY the facts given. "
          "Never invent dates, URLs or numbers. Write dates exactly as YYYY-MM-DD and copy URLs exactly. "
          "Say 'earliest copy found' (never 'original', never 'fake'). Respect the verdict label: for UNVERIFIED or "
          "CONSISTENT_WITH_CLAIM say what was and was not found and never imply the video is fake or recycled. "
          "If a fact says POST DATE ONLY, write exactly 'posted on YYYY-MM-DD; no event date was stated' and never call that date the date of an event. "
          "If a media_nature.summary fact is given, mention it as an AI observation from sampled frames, never as proof. "
          "A visual match proves the footage was published earlier; it does not prove the event did not happen. "
          "LEAD with the strongest finding: the FIRST sentence of every card must state it (priority: a proven earlier copy; the uploader's own AI/animation label; "
          "the input video posted before the claimed event, which is the video's own post date and NEVER an 'earlier copy' or 'match'; agreeing signals). "
          "Never open with the claim or the post title, and never start with 'The claim is'; the claim may appear only after the lead. "
          "Reply with one JSON object only.")


def _prompt(r: dict, langs: list[str]) -> str:
    F = facts(r)
    names = {lg: LANG_NAMES.get(lg, lg) for lg in langs}
    return (f"Facts (field path -> value):\n{json.dumps(F, ensure_ascii=False, indent=1)}\n\n"
            f"Verdict label: {F['final_verdict.label']}\n"
            f"Write the card in these languages: {json.dumps(names)}. Each card is 3 to 5 plain, natural sentences for a social-media reader, same content in each language.\n"
            f"LEAD KIND: {lead_kind(r)}. {LEAD_INSTRUCTION[lead_kind(r)]} The first sentence must be this lead, in every language. Do not start with the claim or the post title.\n"
            "Content to include: the finding above first, then the claim (optional, after the lead); if an earliest copy is in the facts, its date, uploader/platform and URL and (for RECYCLED_FOOTAGE) that it predates the claimed event date; "
            "the match evidence (consistent frames, inliers) when given; and one sentence saying what this does NOT prove. Do not print the label name itself.\n"
            "Return JSON: {\"<lang code>\": [{\"text\": \"one sentence\", \"ref\": \"<one field path from the facts above that the sentence is based on>\"}, ...], ...}\n"
            "Every sentence needs a ref. Use only the language codes requested.")


def _call(model, r, langs):
    txt, cost = llm.chat(model, _prompt(r, langs), max_tokens=8000, system=SYSTEM, timeout=180, baseten=True)
    return llm.parse_json(txt), cost


def generate_debunk(r: dict, chat=None) -> dict:
    """Returns the report["debunk"] dict. `chat(model, r, langs) -> (json, cost)` can be injected for tests."""
    chat = chat or _call
    langs = detect_langs(r)
    cost, errors, used = 0.0, [], None
    attempts = [MODEL, GLIMMER_MODEL, FALLBACK_MODEL]  # Spark Open (Baseten), then Glimmer, then Spark 1.3
    for model in attempts:
        try:
            obj, c = chat(model, r, langs)
            cost += c
            prov = llm.last_provider() if chat is _call else "injected"
        except Exception as e:
            errors.append(f"{model}: {type(e).__name__}: {str(e)[:120]}"); continue
        texts, refs, bad = {}, {}, {}
        for lg in langs:
            sents = [x for x in (obj.get(lg) or []) if isinstance(x, dict)]
            probs = check_grounded(r, sents, lg)
            if probs:
                bad[lg] = probs[:4]
            else:
                texts[lg] = " ".join(x["text"].strip() for x in sents)
                refs[lg] = [{"sentence": x["text"].strip(), "ref": x["ref"]} for x in sents]
        if not bad:
            return dict(texts=texts, field_refs=refs, model=model, provider=prov, grounded=True, languages=langs, cost_usd=round(cost, 5))
        errors.append(f"{model}: grounding failed {json.dumps(bad, ensure_ascii=False)[:300]}")
        used = model
    # fallback: deterministic template (grounded by construction) for en/ne/hi; unknown languages are dropped
    tpl = template_debunk(r, [lg for lg in langs if _template(r, lg) is not None])
    tpl.update(model=f"template (llm rejected: {used})" if used else "template (llm unavailable)", grounded=False,
               languages=list(tpl["texts"]), llm_errors=errors, cost_usd=round(cost, 5))
    return tpl


def validate_debunk(d: dict):
    """Structure check used by validate_report (field is optional; schema 1.3)."""
    assert isinstance(d, dict) and isinstance(d.get("texts"), dict) and d["texts"], "debunk.texts missing"
    assert isinstance(d.get("grounded"), bool) and d.get("model"), "debunk needs model and grounded flag"
    assert isinstance(d.get("field_refs"), dict) and set(d["field_refs"]) == set(d["texts"]), "debunk.field_refs must cover every language"
    for lg, refs in d["field_refs"].items():
        assert refs and all(x.get("ref") and x.get("sentence") for x in refs), f"debunk.field_refs[{lg}] needs a ref per sentence"
    return d


def attach(report_path, chat=None) -> dict:
    """Generate a debunk for the report.json at report_path (str/Path), validate and write it back."""
    from pathlib import Path
    from .report import validate_report
    p = Path(report_path)
    r = json.loads(p.read_text())
    r.pop("debunk", None)
    d = generate_debunk(r, chat)
    # the post-check must hold on what gets stored: verify the stored sentences against the report once more
    for lg, refs in d["field_refs"].items():
        probs = check_grounded(r, [{"text": x["sentence"], "ref": x["ref"]} for x in refs], lg)
        assert not probs, f"stored debunk[{lg}] fails grounding: {probs}"
    r["debunk"] = d
    if "1.2" <= r.get("schema_version", "") < "1.3":      # 1.1 demo reports keep their version (they lack media_nature)
        r["schema_version"] = "1.3"
    validate_report(r)
    p.write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    return d


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        d = attach(a)
        print(a, d["model"], "grounded=" + str(d["grounded"]), d.get("cost_usd"))
