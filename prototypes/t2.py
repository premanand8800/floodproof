from fp_common import *
import sys
doms=["boomlive.in","newschecker.in","thequint.com","cyberpeace.org","factly.in","factcheck.afp.com","altnews.in"]
qs=["old video falsely shared as flood Nepal fact check","old video shared as landslide fact check originally uploaded YouTube","video falsely linked to flash flood Uttarakhand Himachal fact check old video","cyclone video fake old video fact check originally","earthquake video fact check old video Myanmar Afghanistan unrelated","Wayanad landslide old video fact check","Sikkim flood Teesta old video fact check","Pakistan floods 2025 old video fact check","Bhotekoshi Nepal flood August 2026 old video fact check","Cyclone Montha Remal Michaung old video fact check","Dharali Uttarkashi flash flood old video fact check","Assam flood old video fact check false claim","Bangladesh flood old video fact check", "Mumbai Delhi rain flood old video from other country fact check"]
seen={}
for q in qs:
    for d in [None]:
        r=tavily(q,10,domains=doms)
        for x in r.get("results",[]):
            if x["url"] not in seen:
                seen[x["url"]]=x["title"]
json.dump(seen,open("bench/search_hits.json","w"),indent=1)
for u,t in seen.items(): print(u,"|",t[:90])
print(len(seen))
