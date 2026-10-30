"""'How it spread': the proven copies ordered by their earliest independent date, each with a LIKELY parent when the evidence allows.

A parent link is never proof. It is shown as "likely derived, not proof" and rests only on measurable evidence:
  * date order (required: the parent must be dated earlier, by at least a day or by exact timestamps)
  * crop / zoom: the homography from the input frame to each copy frame (from the stored RANSAC inliers) gives the part of the
    input frame each copy shows. A copy whose view lies inside an earlier copy's view and is clearly smaller is a likely crop or
    zoom of it; a copy that shows MORE than another cannot be derived from it (that pair is excluded as parent)
  * an overlay / caption (Muse visible_text) shared by the two copies that no copy dated before the parent shows
  * an @username watermark read by Muse on the later copy that is the earlier copy's uploader
  * the same mirroring (both mirrored relative to the input)
At least one content reason (crop, overlay, username) is required besides date order; mirroring alone is not enough. If two
earlier copies fit equally well, or nothing fits, likely_parent stays null.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

LABEL = "likely derived, not proof"
MIN_PROOF_POINTS = 12
CROP_RATIO = 0.8          # a copy's view must be at most this fraction of the other's to count as a crop/zoom
CONTAIN = 0.85            # and at least this fraction of it must lie inside the other's view
SAME_MOMENT_S = 1.0       # both proof frames must come from the same moment of the input for the views to be comparable
MIN_SCORE = 2


def view_region(proof):
    """The part of the INPUT frame (unit square) that one copy frame shows: the copy's unit square mapped back through the
    homography fitted to the proof inliers. Returns (polygon Nx2, area) or None."""
    inl = (proof or {}).get("inliers") or []
    if len(inl) < MIN_PROOF_POINTS:
        return None
    a = np.float32([[p[0], p[1]] for p in inl]); b = np.float32([[p[2], p[3]] for p in inl])
    H, mask = cv2.findHomography(a, b, cv2.RANSAC, 0.03)
    if H is None or mask is None or int(mask.sum()) < MIN_PROOF_POINTS:
        return None
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    corners = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]).reshape(-1, 1, 2)
    poly = cv2.perspectiveTransform(corners, Hi).reshape(-1, 2)
    if not np.isfinite(poly).all() or not cv2.isContourConvex(poly.astype(np.float32).reshape(-1, 1, 2)):
        return None
    area = float(cv2.contourArea(poly.astype(np.float32)))
    return (poly, area) if 0.02 < area < 20 else None


def _inter(pa, pb):
    area, _ = cv2.intersectConvexConvex(pa.astype(np.float32), pb.astype(np.float32))
    return float(area)


def crop_relation(p_parent, p_child):
    """'crop' if the child's view is a clearly smaller part of the parent's, 'wider' if the child shows clearly MORE than the parent
    (so it cannot derive from it), else None. Needs comparable frames: same moment of the input, same mirroring."""
    if not p_parent or not p_child:
        return None
    if abs(float(p_parent.get("viral_t", 0)) - float(p_child.get("viral_t", 0))) > SAME_MOMENT_S or bool(p_parent.get("mirrored")) != bool(p_child.get("mirrored")):
        return None
    rp, rc = view_region(p_parent), view_region(p_child)
    if not rp or not rc:
        return None
    (pp, ap), (pc, ac) = rp, rc
    inter = _inter(pp, pc)
    if ac <= CROP_RATIO * ap and inter / ac >= CONTAIN:
        return dict(kind="crop", child_area=round(ac, 2), parent_area=round(ap, 2), overlap=round(inter / ac, 2))
    if ap <= CROP_RATIO * ac and inter / ap >= CONTAIN:
        return dict(kind="wider", child_area=round(ac, 2), parent_area=round(ap, 2), overlap=round(inter / ap, 2))
    return None


def _norm(t):
    return re.sub(r"[^\w@]+", " ", str(t).lower(), flags=re.U).strip()


def _texts(read):
    return {_norm(x["text"]) for x in (read or {}).get("visible_text", []) if len(_norm(x["text"])) >= 4}


def _key(e):
    return (e.get("upload_timestamp") or "", e.get("upload_date") or "")


def earlier(a, b):
    """True if copy a is dated before copy b (a whole day earlier, or an exact timestamp earlier), both dates independent."""
    if not (a.get("date_independent") and b.get("date_independent") and a.get("upload_date") and b.get("upload_date")):
        return False
    if a["upload_date"] < b["upload_date"]:
        return True
    ta, tb = a.get("upload_timestamp"), b.get("upload_timestamp")
    return bool(a["upload_date"] == b["upload_date"] and ta and tb and ta < tb)


def parent_evidence(p, c, entries, reads):
    """Reasons (text) and score for 'p is a likely parent of c'. Returns None when p is excluded or has no content evidence."""
    if not earlier(p, c):
        return None
    pp, pc = p.get("proof"), c.get("proof")
    rel = crop_relation(pp, pc)
    if rel and rel["kind"] == "wider":
        return None
    reasons, score = [f"dated earlier ({p['upload_date']} vs {c['upload_date']})"], 0
    if rel:
        score += 2
        reasons.append(f"crop or zoom: this copy shows {rel['child_area']:.0%} of the input frame and the earlier copy {rel['parent_area']:.0%}, "
                       f"and {rel['overlap']:.0%} of this copy's view lies inside the earlier copy's view (homography of the proof frames)")
    rp, rc = reads.get(p["url"]), reads.get(c["url"])
    if rp and rc:
        shared = _texts(rp) & _texts(rc)
        before = [reads[e["url"]] for e in entries if e["url"] in reads and e["url"] != p["url"] and earlier(e, p)]
        added = sorted(t for t in shared if before and all(t not in _texts(b) for b in before))
        if added:
            score += 2
            reasons.append("shared overlay or caption not seen on any earlier copy: " + ", ".join(repr(t) for t in added[:3]) + " (Muse reading, an AI observation)")
    if rc:
        h_up = re.sub(r"[^a-z0-9]", "", str(p.get("uploader") or "").lower())
        url_part = re.sub(r"[^a-z0-9]", "", p["url"].lower())
        for u in rc.get("usernames", []):
            hu = re.sub(r"[^a-z0-9]", "", u.lower())
            if len(hu) >= 4 and ((h_up and (hu == h_up or (len(h_up) >= 5 and hu in h_up))) or ("@" + u.lower() in p["url"].lower())):
                score += 3
                reasons.append(f"this copy shows the watermark @{u}, which is the uploader of the earlier copy (Muse reading, an AI observation)")
                break
    if pp and pc and pp.get("mirrored") and pc.get("mirrored"):
        score += 1
        reasons.append("both copies are mirrored relative to the input")
    return dict(reasons=reasons, score=score) if score >= MIN_SCORE else None


def build_spread(rep):
    """List of spread entries from a report's proven copies (pipeline + snowball), ordered by earliest independent date."""
    v = rep.get("visual_matching") or {}
    proven = v.get("proven_matches") or []
    if not proven:
        return []
    from .report import platform_of
    reads = {m["url"]: m["muse_read"] for m in proven if m.get("muse_read")}
    order = sorted(proven, key=lambda m: (not (m.get("date_independent") and m.get("upload_date")), m.get("upload_date") or "9999",
                                          m.get("upload_timestamp") or "", m["url"]))
    out = []
    for c in order:
        best, note = None, None
        if c.get("upload_date") and c.get("date_independent"):
            cands = []
            for p in order:
                if p is c:
                    continue
                ev = parent_evidence(p, c, order, reads)
                if ev:
                    cands.append((ev["score"], p, ev))
            if cands:
                top = max(s for s, _, _ in cands)
                tops = [x for x in cands if x[0] == top]
                if len(tops) == 1:
                    _, p, ev = tops[0]
                    best = dict(url=p["url"], platform=platform_of(p["url"]), label=LABEL, reasons=ev["reasons"], score=ev["score"])
                else:
                    note = f"the evidence fits {len(tops)} earlier copies equally well, so no single likely parent is named"
        entry = dict(url=c["url"], platform=platform_of(c["url"]), uploader=c.get("uploader"), date=c.get("upload_date"),
                     date_source=c.get("date_source"), date_independent=bool(c.get("date_independent")),
                     found_by=c.get("found_by") or c.get("adapter"), likely_parent=best)
        if c.get("upload_timestamp"):
            entry["upload_timestamp"] = c["upload_timestamp"]
        if note:
            entry["parent_note"] = note
        mr = reads.get(c["url"])
        if mr and mr.get("visible_text"):
            entry["visible_text"] = [x["text"] for x in mr["visible_text"]][:6]
        out.append(entry)
    return out


def validate_spread(r):
    """Optional report['spread']: ordered by date, every url a proven copy, every parent strictly earlier with text reasons."""
    sp = r["spread"]
    assert isinstance(sp, list), "spread must be a list"
    proven = {m["url"]: m for m in (r.get("visual_matching") or {}).get("proven_matches", [])}
    assert len(sp) == len(proven) and {e.get("url") for e in sp} == set(proven), "spread must list exactly the proven copies"
    seen, prev = {}, None
    for e in sp:
        m = proven[e["url"]]
        assert e.get("date") == m.get("upload_date") and e.get("date_source") == m.get("date_source"), "spread date must be the proven copy's date"
        assert e.get("found_by"), "spread entry needs found_by"
        k = (e.get("date") is None, e.get("date") or "")
        assert prev is None or k >= prev, "spread must be ordered by date"
        prev = k
        seen[e["url"]] = e
        lp = e.get("likely_parent")
        if lp is not None:
            par = seen.get(lp.get("url"))
            assert par is not None and lp["url"] != e["url"], "likely_parent must be an earlier spread entry"
            assert lp.get("label") == LABEL, "likely_parent must be labelled 'likely derived, not proof'"
            assert isinstance(lp.get("reasons"), list) and len(lp["reasons"]) >= 2 and all(isinstance(x, str) and x for x in lp["reasons"]), \
                "likely_parent needs date order plus at least one content reason"
            assert e.get("date_independent") and par.get("date_independent") and par["date"] and e["date"] and par["date"] <= e["date"], \
                "likely_parent needs independent dates with the parent earlier"


def validate_snowball(r):
    """Optional report['snowball']: capped, every new copy matcher-proven and listed in proven_matches."""
    from .snowball import MAX_ROUNDS, MAX_QUERIES, MAX_EXTRA_VERIFY, MAX_SECONDS
    s = r["snowball"]
    assert isinstance(s, dict) and isinstance(s.get("rounds"), list) and len(s["rounds"]) <= MAX_ROUNDS, "snowball.rounds invalid"
    assert str(s.get("stopped_reason") or "").strip(), "snowball needs stopped_reason"
    assert isinstance(s.get("seconds"), (int, float)) and not isinstance(s["seconds"], bool) and s["seconds"] <= MAX_SECONDS + 15, "snowball.seconds invalid"
    c = s.get("cost")
    assert isinstance(c, dict) and all(isinstance(c.get(k), (int, float)) for k in ("openrouter_usd", "lens_calls", "verifications")), "snowball.cost invalid"
    proven = {m["url"]: m for m in (r.get("visual_matching") or {}).get("proven_matches", [])}
    total = 0
    for i, rd in enumerate(s["rounds"], 1):
        assert rd.get("round") == i and rd.get("seed_copy") in proven, "snowball round needs its number and a proven seed copy"
        assert isinstance(rd.get("queries"), list) and len(rd["queries"]) <= MAX_QUERIES, "snowball round has too many queries"
        for q in rd["queries"]:
            assert str(q.get("q") or "").strip() and str(q.get("rationale") or "").strip() and q.get("source_field"), "snowball query needs q, rationale, source_field"
        assert isinstance(rd.get("verified"), int) and not isinstance(rd["verified"], bool) and rd["verified"] >= 0, "snowball round verified must be a count"
        total += rd["verified"]
        for u in rd.get("proven_new", []):
            m = proven.get(u)
            assert m is not None, "a snowball copy must be a matcher-proven copy listed in proven_matches"
            assert str(m.get("found_by") or "").startswith(f"snowball round {i}"), "snowball copy must be labelled with its round"
    assert total <= MAX_EXTRA_VERIFY and total == c["verifications"], "snowball verification count invalid or over the cap"
