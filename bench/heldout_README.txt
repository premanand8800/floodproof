FloodProof HELD-OUT benchmark (heldout.jsonl, H01-H10; media in data/heldout/)
FROZEN. Never tune on this set (no threshold, prompt, or pipeline changes informed by these cases).
Built 2026-10-30. Every case: debunked recycled disaster video (flood/landslide/quake/storm), NEW vs bench/cases.jsonl C01-C35.

How built: fact-check article = lead only. Viral clip = downloadable post cited by the fact-check
(X / Facebook / Instagram / TikTok), saved <=360p, <=60 s via yt-dlp to data/heldout/Hxx.mp4.
Original = footage named by the fact-check (YouTube/IG/FB/X) downloaded temporarily (<=120 s), then deleted
(NOT kept in data/, never in data/memory). Ground truth validated by floodproof.matcher
(ClipFeatures.from_path + match(coarse=12)): only decision==True kept; consistent_frames recorded.
Weak matches (<=4 frames): H04 (2), H06 (3), H10 (4); all decision True.
original_date = platform date from yt-dlp (probe_url); agrees (+-1 day) with the fact-check in all 10 -> label_confirmed=true.
confirmation="platform_metadata": platform upload date and uploader confirmed by yt-dlp; fact-check text was
read via WebFetch summaries (not a full manual page read).
Note: where the original is a news upload of footage filmed earlier, original_date is the earliest platform upload
found, see earliest_date_seen / notes.

Outlets: Newschecker Nepal (5), CyberPeace (3), Fact Crescendo (2). Regions of claim: Nepal, Myanmar/Thailand, UAE, Canada, Assam/India.
Selection bias: outlets whose viral posts are downloadable (TikTok/X/IG/FB links). Many TikTok links were IP-blocked
and many candidates (Quint, AFP, BOOM, Factly: archive-only or 403) were skipped; AFP/BOOM/Quint/Factly/Alt News absent.

Exclusions: anything already in C01-C35 (Dharali/Uttarkashi 2025, Nalagarh, Sainj, Wayanad 2024, Tauktae, Dana, Derna, Sikkim Austria,
Leigh Coates glacier, ...). Cases dropped for failed viral download or matcher False: Newschecker Nepal Bihar/Sendong (0 frames),
Quint Karachi/Kerala, Dismislab Ctg, Kerala->Assam (FC), J&K->Assam (FC), Typhoon Bavi/Dolphin (FC), Nashik, Japan Noto (FC), Myanmar CCTV->Japan (FC).
Several viral clips are 2026 Nepal/Assam flood claims; ground truth is footage identity only.
