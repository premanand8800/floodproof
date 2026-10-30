"""Live eval of floodproof.physical. Run: PYTHONPATH=src uv run python bench/physical_eval.py"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from floodproof import physical  # noqa: E402

MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
KW = r"(flash|flood|floods|GLOF|cloudburst|landslide|cyclone|broken|disaster|mudslide|-Tibet)"

FIXED = [
    ("a Kathmandu Bagmati", "Bagmati River, Kathmandu, Nepal", "2024-09-28", "SUPPORTS"),
    ("b Rasuwa Bhotekoshi", "Rasuwa, Nepal", "2026-08-26", "SUPPORTS"),
    ("c dry control", "Kathmandu, Nepal", "2025-01-15", "NO_EVIDENCE/CONTRADICTS"),
]


def parse_claim(txt):
    m = re.match(r"(.+?)\s+(?:Nepal-Tibet\s+)?" + KW, txt) if False else re.match(r"(.+?)\s+" + KW, txt)
    place = None
    if m:
        place = m.group(1).replace("(", "").replace(")", "").strip()
    place = (place or "").replace("Cyclone Ditwah", "").replace("Cyclone Dana", "").replace("Cyclone hitting", "").strip()
    if txt.startswith("Cyclone Ditwah Chennai"):
        place = "Chennai"
    if txt.startswith("Cyclone Dana"):
        place = "Odisha"
    if txt.startswith("Nepal-Tibet"):
        place = "Nepal"
    dm = re.search(r"(?:(\w{3}) )?(\d{1,2} )?\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:(\d{1,2}) )?(20\d\d)", txt)
    if not dm or not place:
        return None
    day = int(dm.group(4)) if dm.group(4) else None
    return place, f"{dm.group(5)}-{MONTHS[dm.group(3)]:02d}-{day or 15:02d}", day is None


def row(label, place, date, expect, window):
    try:
        r = physical.check(place=place, date=date, window_days=window)
    except Exception as e:  # noqa: BLE001
        return [label, place, date, expect, "EXC", str(e)[:60]]
    hy = r["hydrology"]
    rd, rn = hy.get("river_discharge") or {}, hy.get("rain") or {}
    ev = [a["code"] for a in r["official_event"]["activations"]]
    return [label, place, date, expect, r["verdict"],
            f"ems={ev or r['official_event']['status']} ratio={rd.get('ratio')} peak={rd.get('peak')} "
            f"rain={rn.get('max_mm')} loc={r['location']['confidence']}"]


def main():
    rows = [row(*f, 3) for f in FIXED]
    for line in (ROOT / "bench" / "cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        p = parse_claim(c["claimed_event"])
        if not p:
            rows.append([c["id"], "-", "-", "-", "SKIP", c["claimed_event"]])
            continue
        place, date, month_only = p
        rows.append(row("d " + c["id"], place, date, "-", 15 if month_only else 3))
    print("| case | place | date | expect | verdict | detail |\n|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |", flush=True)


if __name__ == "__main__":
    main()
