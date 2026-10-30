FloodProof GLOBAL benchmark (global.jsonl: G01-G10 debunked, N01-N03 genuine; media in data/global/, gitignored *.mp4)
FROZEN. Never tune on this set. Built 2026-10-30. Goal: test whether the fixed pipeline (system B) works outside South Asia.
Disjoint from bench/cases.jsonl C01-C35 and bench/heldout.jsonl H01-H10 by URL (checked in global_unseen_check.json) and by keyframe
matching against EVERY memory entry (exhaustive). Memory is read-only in this run (remember=False); these cases are NOT added to memory.

How built: fact-check article = lead only (read through WebFetch summaries, not a manual page read). Viral clip = a public post the
fact-check cites (X / Instagram), downloaded by yt-dlp without login, <=360p, first <=60 s, to data/global/. Public pages only.
Originals named by the fact-check were downloaded to a scratch dir, compared with floodproof.matcher (match(coarse=12)), then deleted.
matcher_validated: true = decision True (frames in matcher_consistent_frames); false = original downloaded or reachable but the matcher did
NOT confirm (altered / mirrored clip, or original not downloadable), ground truth then rests on the fact-check text only; null = no original exists.
Dates: original_date = platform upload date of the named original URL when yt-dlp could read it, else the fact-check's date.

Regions (claim place / origin): Europe G01; Americas G02, G03; Africa G04, G10; Middle East G05; East Asia/Russia Far East G06, G09;
Southeast Asia G07; Oceania G08 (origin Auckland, claim NYC). 6 regions, 8 outlets-cases from Full Fact (5), PolitiFact (2), Fact Crescendo (2), AAP (1).
Outlet bias: only fact-checks that cite an X or Instagram post that yt-dlp could still fetch. AFP/Reuters/Maldita/Lupa/Africa Check/Vera Files
appeared in search but were skipped: Africa Check 403 to fetch, Vera Files/AAP/Maldita/Lupa viral posts were archived Facebook/mvau.lt links (not
downloadable), AFP pages unreachable. Many Facebook reels and TikToks failed ("Cannot parse data", IP blocked, permission).
Exclusions after download: Wyoming elk (viral X clip is a different night-fire clip than the cited original, matcher 0), Thailand-2004/Japan tsunami X (suspended),
NZ Gabrielle X (media 403), Dubai/Japan tsunami, Poland/Spain, Quezon City, Athens-viral-FB etc. (no downloadable viral post).
Caveats: G01/G03/G08/G10 not matcher-validated; G03 and G10 share one original; G07 is a partly genuine compilation (3 of 6 clips real Indonesia);
G09 is AI-generated with no original; G06 original_date is a later re-upload of Newsflare footage (true first upload has no URL).
Genuine clips N01-N03 (Jerusalem wildfire 2025-04-30, Hurricane Beryl Carriacou 2024-07-01, Durban waves 2017-03-12): claim and date are correct.

UNSEEN CHECK (run before any B run; bench/global_unseen_check.json): G01 and G04 are NOT unseen and are excluded from scoring by design (visible in the JSON).
G04: its viral clip visually matches memory entry youtube v063it_3tTA, which is one of the Full Fact Apr-2016 Saudi flood originals already indexed from an earlier benchmark case.
G01: matches 10 memory entries (Wayanad/Sainj/Tauktae etc.), most likely spurious matches of a low-feature clip, but the rule is applied as pre-registered, not adjusted.
Scored set therefore: 8 debunked (G02, G03, G05-G10) + 3 genuine (N01-N03).
