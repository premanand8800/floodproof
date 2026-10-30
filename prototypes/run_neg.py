import glob,json,time,itertools,os
from matcher import *
def vp(k):
    g=[x for x in glob.glob(f"data/videos/{k}.*") if not x.endswith(".part")]; return g[0] if g else None
cache={}
def P(k):
    if k not in cache: cache[k]=Prep(vp(k))
    return cache[k]
# negatives: different footage; hard negatives = same genre (floods / muddy water / rivers), incl. same country or same source
NEG=[("ig_leigh","yt_ie_nalagarh"),("ig_sainj","ig_pak2022"),("ig_viral_italy","ig_dharali_accu"),("yt_ie_nalagarh","yt_sun_italy"),
("x_iambmane","yt_croc"),("yts_nepal_vanTV","yts_viral_nalagarh"),("yt_dawn","yt_tauktae1"),("yt_hum","yts_sainj"),("ig_wayanad_el","yt_tribune_nalagarh"),
("yt_leigh","ig_viral_italy"),("ig_pak2022","yt_sun_italy"),("yts_ajesh_buddha","ig_dharali_accu"),("x_dmuppavarapu","yt_dawn"),("yt_tauktae2","yt_wayanad_el"),("ig_dharali_accu","yts_sainj"),("yt_hum","yt_ie_nalagarh")]
rows=[]
for a,b in NEG:
    if not vp(a) or not vp(b): print("skip",a,b); continue
    A,B=P(a),P(b); r=compare(A,B)
    row=dict(a=a,b=b,verdict=verdict(r),consistent=r["consistent"],max_inliers=r["max_inliers"],n_pairs=r["n_pairs"],best_phash_dist=phash_best(A,B))
    if r["pairs"]: row["png"]=evidence_png(A,B,r,f"evidence/neg_{a}__{b}.png",f"NEG {a} vs {b}")
    print(row,flush=True); rows.append(row)
json.dump(rows,open("bench/neg_pairs.json","w"),indent=1)
# synthetic recycled variants of a real original (ig_sainj + yt_leigh): crop, mirror, overlay, rescale+reencode, combo
os.makedirs(".cache/var",exist_ok=True)
V={"crop70":"crop=iw*0.7:ih*0.7:iw*0.15:ih*0.2,scale=360:-2",
   "mirror":"hflip",
   "overlay":"drawbox=x=0:y=ih*0.75:w=iw:h=ih*0.2:color=black@0.9:t=fill,drawtext=text='BREAKING NEWS NEPAL FLOOD 2026':fontcolor=white:fontsize=18:x=10:y=h*0.8",
   "lowres":"scale=180:-2,scale=360:-2,eq=contrast=1.2:saturation=0.8",
   "combo":"hflip,crop=iw*0.8:ih*0.8:iw*0.1:ih*0.1,scale=300:-2,drawbox=x=0:y=0:w=iw:h=24:color=red@0.8:t=fill"}
vrows=[]
for src in ("yt_leigh","ig_sainj"):
    A=P(src)
    for n,vf in V.items():
        dst=f".cache/var/{src}_{n}.mp4"
        try: make_variant(vp(src),dst,vf)
        except Exception as e: print("ffmpeg fail",n); continue
        B=Prep(dst); r=compare(A,B)
        row=dict(src=src,variant=n,verdict=verdict(r),consistent=r["consistent"],max_inliers=r["max_inliers"],flip=r["flip"],best_phash_dist=phash_best(A,B))
        if r["pairs"]: row["png"]=evidence_png(A,B,r,f"evidence/var_{src}_{n}.png",f"VARIANT {src} {n}")
        print(row,flush=True); vrows.append(row)
json.dump(vrows,open("bench/variants.json","w"),indent=1)
