import sys,os,json,glob,time
from matcher import *
def vp(k):
    g=glob.glob(f"data/videos/{k}.*"); return g[0] if g else None
cache={}
def P(k):
    if k not in cache: cache[k]=Prep(vp(k))
    return cache[k]
TRUE=[("T1_nepalGLOF_x_vs_ig","x_iambmane","ig_leigh"),
("T2_nepalGLOF_x2_vs_yt","x_dmuppavarapu","yt_leigh"),
("T3_nalagarh_short_vs_IE","yts_viral_nalagarh","yt_ie_nalagarh"),
("T4_nalagarh_IE_vs_tribune","yt_ie_nalagarh","yt_tribune_nalagarh"),
("T5_italy_ig_vs_sun","ig_viral_italy","yt_sun_italy"),
("T6_sainj_ig_vs_yts","ig_sainj","yts_sainj"),
("T7_wayanad_ig_vs_yt","ig_wayanad_el","yt_wayanad_el"),
("T8_dawn_vs_hum","yt_dawn","yt_hum"),
("T9_tauktae_a_vs_b","yt_tauktae1","yt_tauktae2"),
]
sel=sys.argv[1] if len(sys.argv)>1 else "true"
rows=[]
for name,a,b in TRUE:
    if not vp(a) or not vp(b): print("skip",name); continue
    t=time.time(); A,B=P(a),P(b); r=compare(A,B)
    ph=phash_best(A,B)
    png=evidence_png(A,B,r,f"evidence/{name}.png",f"{name}: consistent={r['consistent']} maxinl={r['max_inliers']} offset={r['offset']}") if r["pairs"] else None
    row=dict(name=name,kind="true",verdict=verdict(r),score=r["score"],consistent=r["consistent"],max_inliers=r["max_inliers"],n_pairs=r["n_pairs"],offset=r["offset"],flip=r["flip"],best_phash_dist=ph,png=png,sec=round(time.time()-t,1))
    print(row); rows.append(row)
json.dump(rows,open("bench/true_pairs.json","w"),indent=1)
