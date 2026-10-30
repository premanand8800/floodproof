import sys,json,glob,os,re,time,concurrent.futures as cf
from urllib.parse import urlparse,parse_qs
from search_loop import *
CASES['V1']=('V1var_combo','2026-08-26','synthetic crop+mirror+overlay of Leigh Columbia Glacier clip',['DcUFQZhvGzS'])   # CASES, info, dl, Prep, compare...
HOST=("youtube.com/watch","youtube.com/shorts","youtu.be","instagram.com/reel","instagram.com/p/","facebook.com/","x.com/","twitter.com/","tiktok.com","dailymotion.com/video","vimeo.com")
def clean(u): return re.sub(r"[?&]utm_[^&]*","",u).rstrip("?&")
def cands_for(case,allnews=True):
    seen={}
    for f in glob.glob(f".cache/rev/yx_*.json"):
        d=json.load(open(f))
        for k,v in d.items():
            if not k.startswith(case+"_"): continue
            for u,t in v.get("links",[]):
                if "ytimg.com" in u or re.search(r"\.(jpe?g|png|webp)(\?|$)",u) or "googleusercontent" in u: continue
                if u.startswith("http") and len(t)>12 and not t.endswith(".com"):
                    seen.setdefault(clean(u),(t,k))
    return seen
def run(case):
    key,qdate,claim,truth=CASES[case]; qd=qdate.replace("-","")
    A=Prep(glob.glob(f"data/videos/{key}.*")[0]); c=cands_for(case)
    vid={u:v for u,v in c.items() if any(h in u for h in HOST)}
    print(case,"cands",len(c),"video-host",len(vid),flush=True)
    def work(it):
        u,(title,src)=it
        inf=info(u)
        if not inf: return dict(url=u,title=title,src=src,status="no video/unavailable")
        d,du=inf
        if d not in("NA","None") and d>qd: return dict(url=u,title=title,src=src,date=d,status="later than query")
        try:
            if du not in("NA","None") and float(du)>600: return dict(url=u,title=title,date=d,src=src,status="too long")
            dur=float(du) if du not in("NA","None") else 60
        except: dur=60
        os.makedirs(f".cache/rv/{case}",exist_ok=True)
        f=dl(u,f".cache/rv/{case}/{abs(hash(u))%10**8}",sec=60 if dur<=120 else 150,h=360 if dur<=120 else 240)
        if not f: return dict(url=u,title=title,date=d,src=src,status="download failed")
        try: B=Prep(f,maxdur=150,fps=2.0 if dur>120 else 3.0,maxframes=400); r=compare(A,B)
        except Exception as e: return dict(url=u,date=d,status="match error")
        png=None
        if verdict(r): png=evidence_png(A,B,r,f"evidence/rev_{case}_{abs(hash(u))%10**6}.png",f"{case} yandex-rev cand {u[-45:]} date={d} cons={r['consistent']} inl={r['max_inliers']}")
        os.remove(f)
        return dict(url=u,title=title,date=d,src=src,status="checked",match=verdict(r),consistent=r["consistent"],max_inliers=r["max_inliers"],png=png,is_truth=any(t in u for t in truth))
    # video-host candidates plus news pages (yt-dlp generic probe), cap
    items=list(vid.items())[:14]
    others=[(u,v) for u,v in c.items() if u not in vid][:8]
    with cf.ThreadPoolExecutor(4) as ex: res=[x for x in ex.map(work,items+others) if x]
    hits=[x for x in res if x.get("match")]
    out=dict(case=case,n_cands=len(c),n_video=len(vid),results=res,hits=hits)
    json.dump(out,open(f"bench/rev_{case}.json","w"),indent=1,default=str)
    print(case,"HITS",[(h["url"],h["date"],h["consistent"],h["max_inliers"],h.get("is_truth")) for h in hits],flush=True)
if __name__=="__main__":
    for c in sys.argv[1:]: run(c)
