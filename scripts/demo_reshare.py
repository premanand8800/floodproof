"""Memory re-share demo: "Fakes recycle. So does our memory."

1. take the demo success clip (data/videos/x_iambmane.mp4, the source of demo/success_recycled),
2. make a re-share variant with ffmpeg: crop to 80%, mirror, Nepali caption bar, re-encode at a much lower bitrate,
3. run investigate() on the variant with memory ON and external search OFF (no Yandex/Tavily/YouTube/Lens/Vision, no Muse call),
   against a COPY of data/memory (the real memory is never written), and time it.
The first-run time (full pipeline with external search + Muse) is read from demo/success_recycled/report.json.

usage: uv run python scripts/demo_reshare.py
writes: .cache/reshare/variant.mp4, bench/reshare_demo.json, demo/reshare_memory/report.json (+ evidence/)
"""
import json, shutil, subprocess, sys, time
from pathlib import Path
from floodproof import ROOT, load_env
from floodproof.memory import Memory, DEFAULT_DIR
from floodproof import pipeline
from floodproof.report import validate_report

load_env()
SRC = ROOT / "data" / "videos" / "x_iambmane.mp4"
WORK = ROOT / ".cache" / "reshare"
VARIANT = WORK / "variant.mp4"
CAPTION = "बाढीको भिडियो: रासुवा, आजै"     # "Flood video: Rasuwa, today"
CAPFILE = WORK / "caption.txt"
FONT = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf"
FIRST_RUN = ROOT / "demo" / "success_recycled" / "report.json"


def probe(p):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,bit_rate",
                        "-show_entries", "format=bit_rate,size", "-of", "json", str(p)], capture_output=True, text=True, check=True).stdout
    j = json.loads(o); s = j["streams"][0]
    return dict(width=s["width"], height=s["height"], bit_rate=int(s.get("bit_rate") or j["format"]["bit_rate"]), size_bytes=int(j["format"]["size"]))


def make_variant():
    WORK.mkdir(parents=True, exist_ok=True)
    CAPFILE.write_text(CAPTION)      # textfile avoids ffmpeg escaping of ':' in the Nepali text
    # crop centre 80%, mirror, caption bar (opaque box at the bottom, Nepali text), re-encode at ~150 kb/s with low-quality AAC
    vf = ("crop=iw*0.8:ih*0.8,hflip,scale=trunc(iw/2)*2:trunc(ih/2)*2,"
          f"drawbox=x=0:y=ih-ih*0.12:w=iw:h=ih*0.12:color=black@0.8:t=fill,"
          f"drawtext=fontfile={FONT}:textfile={CAPFILE}:fontcolor=white:fontsize=h*0.036:x=(w-text_w)/2:y=h-h*0.12+(h*0.12-text_h)/2")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(SRC), "-vf", vf, "-c:v", "libx264", "-b:v", "150k", "-maxrate", "150k",
                    "-bufsize", "300k", "-preset", "fast", "-c:a", "aac", "-b:a", "48k", str(VARIANT)], check=True)


def main():
    make_variant()
    ob, vb = probe(SRC), probe(VARIANT)
    mem_copy = WORK / "memory_copy"
    if not mem_copy.exists():
        shutil.copytree(DEFAULT_DIR, mem_copy)
    mem = Memory(mem_copy)
    n_before = mem.stats()["entries"]
    out = ROOT / "demo" / "reshare_memory"; shutil.rmtree(out, ignore_errors=True)
    t = time.time(); marks = []

    def log(msg):
        marks.append((round(time.time() - t, 1), msg)); print(f"[{marks[-1][0]:6.1f}s] {msg}", flush=True)
    rep = pipeline.investigate(str(VARIANT), claim="Nepal glacier lake outburst flood, August 2026", date="2026-08-26", place="Nepal",
                               out_dir=out, memory=mem, remember=False, log=log,
                               use_muse=False, use_yandex=False, use_tavily=False, use_gvision=False, use_youtube=False, use_lens=False)
    wall = round(time.time() - t, 1)
    validate_report(rep)
    first = json.loads(FIRST_RUN.read_text())
    m = rep["memory"]; vm = rep["visual_matching"]
    res = dict(
        demo="memory re-share: crop 80% + mirror + Nepali caption bar + low-bitrate re-encode",
        original=dict(path="data/videos/x_iambmane.mp4", **ob), variant=dict(path=str(VARIANT.relative_to(ROOT)), caption=CAPTION, **vb),
        memory_mode="data/memory copy (37 entries, read-only copy; remember=False)", memory_entries_before=n_before, memory_entries_after=mem.stats()["entries"],
        external_search="OFF (yandex, tavily, youtube api, lens, vision); Muse OFF (no paid call)",
        rerun=dict(wall_s=wall, report_timing_s=rep.get("timing_s"), memory_status=m.get("status"), memory_lookup_s=m.get("seconds"), memory_mode=m.get("mode"),
                   verdict=rep["final_verdict"]["label"], n_proven=len(vm["proven_matches"]),
                   proven=[dict(url=p["url"], adapter=p["adapter"], independent_date=p["independent_date"], date_independent=p["date_independent"],
                                match=p["match"]) for p in vm["proven_matches"]]),
        first_run=dict(wall_s=first.get("timing_s"), memory_status=first["memory"].get("status"), verdict=first["final_verdict"]["label"],
                       note="full pipeline: memory + Muse + Yandex + Tavily + physical, from demo/success_recycled/report.json"),
        phase_log=marks, speedup=round(first.get("timing_s", 0) / wall, 1) if wall else None, cost_usd=0.0)
    (ROOT / "bench" / "reshare_demo.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("rerun", "first_run", "speedup")}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
