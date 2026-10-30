"""floodproof CLI:  investigate <file-or-url> --claim ... --date YYYY-MM-DD   |   memory seed|stats|add"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="floodproof")
    sp = ap.add_subparsers(dest="cmd", required=True)
    inv = sp.add_parser("investigate", help="investigate a clip (local file or URL)")
    inv.add_argument("source")
    inv.add_argument("--claim"); inv.add_argument("--date", help="claimed event date YYYY-MM-DD"); inv.add_argument("--place")
    inv.add_argument("--claimed-source", help="URL claimed to be the original (tracked separately from proven copies)")
    inv.add_argument("--out", help="output dir (default runs/<id>)")
    inv.add_argument("--no-yandex", action="store_true"); inv.add_argument("--no-tavily", action="store_true")
    inv.add_argument("--no-muse", action="store_true"); inv.add_argument("--no-memory", action="store_true")
    inv.add_argument("--remember", action="store_true", help="add the input to memory as discovered:<id>")
    inv.add_argument("--json", action="store_true", help="print report.json instead of the text summary")
    mm = sp.add_parser("memory"); ms = mm.add_subparsers(dest="mcmd", required=True)
    ms.add_parser("seed"); ms.add_parser("stats")
    ad = ms.add_parser("add"); ad.add_argument("source"); ad.add_argument("--url"); ad.add_argument("--date")
    ad.add_argument("--context", default="{}", help="JSON event context"); ad.add_argument("--investigation-id")
    a = ap.parse_args(argv)

    from . import load_env
    load_env()
    if a.cmd == "investigate":
        from .pipeline import investigate
        from .report import render_text
        rep = investigate(a.source, a.claim, a.date, a.place, a.claimed_source, a.out, use_memory=not a.no_memory,
                          use_yandex=not a.no_yandex, use_tavily=not a.no_tavily, use_muse=not a.no_muse,
                          remember=a.remember, log=lambda s: print(s, file=sys.stderr))
        print(json.dumps(rep, indent=2, ensure_ascii=False, default=str) if a.json else render_text(rep))
        from pathlib import Path
        print(f"\nreport: {Path(a.out or 'runs/' + rep['investigation_id']).resolve()}/report.json", file=sys.stderr)
        return
    from .memory import Memory, seed_benchmark
    mem = Memory()
    if a.mcmd == "seed":
        print(seed_benchmark(mem)); print(mem.stats())
    elif a.mcmd == "stats":
        print(json.dumps(mem.stats(), indent=2))
    elif a.mcmd == "add":
        from . import media
        from .matcher import ClipFeatures
        clip = media.fetch(a.source)
        eid = mem.add(ClipFeatures.from_path(clip.path), a.url or clip.url, a.date or clip.upload_date, json.loads(a.context),
                      a.investigation_id, clip.uploader, clip.title, "manual" if not a.investigation_id else "investigated_input")
        media.cleanup(clip); print("added entry", eid)


if __name__ == "__main__":
    main()
