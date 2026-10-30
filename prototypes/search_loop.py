import sys,os,json,glob,base64,re,subprocess,time,concurrent.futures as cf
import cv2
from fp_common import *
from matcher import *
MODEL="meta/muse-spark-1.3"
CASES={ # key: (query video key, query upload date, claim, set of ground-truth original ids/keys)
"S1":("x_iambmane","2026-08-26","Video of glacier lake outburst / flood destroying Nepal-Tibet area, August 2026",["DcUFQZhvGzS","E5TOaoetGDo"]),
"S2":("x_dmuppavarapu","2026-08-27","Video of lake embankment breach causing Nepal-Tibet floods, August 2026",["E5TOaoetGDo","DcUFQZhvGzS"]),
"S3":("yts_viral_nalagarh","2025-08-06","Men stranded in river during Uttarkashi cloudburst, August 2025",["gG8y--iMl04","MBLVrGHbJAc"]),
"S5":("ig_viral_italy","2025-08-05","Muddy flash flood in a canal during Uttarakhand flash flood, August 2025",["Nsb9eSzAEgQ"]),
"S6":("yts_sainj","2024-09-09","Flooding in Himachal Pradesh, 2024",["C_pHwZgvMXg"]),
"S7":("yt_wayanad_el","2024-04-03","Elephants running to safety before Wayanad landslide, July 2024",["C2AMQyYpf_f"]),
"S8":("yt_hum","2024-08-08","Bridge and river flooding, Pakistan, August 2024",["GlbjFxMUs6U"]),
"S9":("yt_tauktae2","2021-05-18","Cyclone Tauktae destruction footage, May 2021",["bSkLZSRyRu0"]),
}
cost={"or":0.0,"tav":0}
def muse(key,claim):
    P=Prep(glob.glob(f"data/videos/{key}.*")[0])
    n=len(P.kf); idx=sorted(set(int(round(i)) for i in np.linspace(0,n-1,min(6,n))))
    SCHEMA='Return ONLY JSON: {"scene":str,"visible_text":[str],"guess_location":str,"queries":[8 strings]}'
    txt0=f"These are {len(idx)} keyframes from a viral disaster video. It is being shared with this claim: '{claim}'. The claim may be false: the footage may be recycled from an EARLIER event, possibly another country. Do NOT trust the claim. Describe what is really visible (terrain, architecture, vehicles, signs, visible text/watermarks/logos, language, season, people) and propose search queries that would find the earliest upload or the true original (news report, YouTube, Instagram, Facebook). Include queries in the likely local language(s) of the real location and neutral English ones that avoid the claimed event. "+SCHEMA
    content=[{"type":"text","text":txt0}]
    for i in idx:
        ok,buf=cv2.imencode(".jpg",P.kf[i][1],[cv2.IMWRITE_JPEG_QUALITY,70])
        content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(buf).decode()}})
    r=requests.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":f"Bearer {OR}"},json={"model":MODEL,"max_tokens":4500,"messages":[{"role":"user","content":content}]},timeout=180).json()
    txt=r["choices"][0]["message"]["content"]
    m=re.search(r"\{.*\}",txt,re.S)
    return json.loads(m.group(0)), r.get("usage",{})
HOSTS=("youtube.com/watch","youtube.com/shorts","youtu.be","instagram.com/reel","instagram.com/p/","facebook.com/watch","facebook.com/reel","x.com/","twitter.com/","tiktok.com","dailymotion.com/video")
def find_cands(queries):
    seen={}
    for q in queries[:6]:
        for kw in ({"include_domains":["youtube.com"]},{"include_domains":["instagram.com","facebook.com","x.com","tiktok.com"]},{"topic":"news"},{}):
            try: r=tavily(q,6,**kw)
            except Exception as e: continue
            cost["tav"]+=1
            for x in r.get("results",[]):
                u=x["url"]
                if any(h in u for h in HOSTS) and u not in seen: seen[u]=x.get("title","")
    return seen
def info(u):
    r=subprocess.run(["uv","run","yt-dlp","--simulate","--no-warnings","-q","--print","%(upload_date)s|%(duration)s",u],capture_output=True,text=True,timeout=90)
    if r.returncode!=0 or not r.stdout.strip(): return None
    l=r.stdout.strip().splitlines()[0]; d,du=l.split("|")
    return d,du
def dl(u,out,sec=60,h=360):
    subprocess.run(["uv","run","yt-dlp","-q","--no-warnings","-f",f"bv*[height<={h}][vcodec!^=av01]/b[height<={h}][vcodec!^=av01]/worst[vcodec!^=av01]","--download-sections","*0-"+str(sec),"--merge-output-format","mp4","--max-filesize","25M","-o",out+".%(ext)s",u],capture_output=True,text=True,timeout=200)
    g=glob.glob(out+".*"); return g[0] if g else None
def run(case):
    key,qdate,claim,truth=CASES[case]
    t0=time.time(); tav0=cost["tav"]
    plan,usage=muse(key,claim)
    c_or=usage.get("cost") or (usage.get("prompt_tokens",0)*1.25e-6+usage.get("completion_tokens",0)*4.25e-6)
    cands=find_cands(plan["queries"])
    A=Prep(glob.glob(f"data/videos/{key}.*")[0])
    qd=qdate.replace("-","")
    res=[]
    def work(item):
        u,title=item
        if key.split("_")[-1] in u: return None
        inf=info(u)
        if not inf: return dict(url=u,title=title,status="unavailable")
        d,du=inf
        try:
            if du not in("NA","None") and float(du)>900: return dict(url=u,date=d,status="too long")
        except: pass
        if d not in("NA","None") and d>qd+"": 
            return dict(url=u,date=d,status="later than query (skipped)")
        os.makedirs(f".cache/cands/{case}",exist_ok=True)
        dur=float(du) if du not in("NA","None") else 60
        f=dl(u,f".cache/cands/{case}/{abs(hash(u))%10**8}",sec=60 if dur<=120 else 240,h=360 if dur<=120 else 240)
        if not f: return dict(url=u,date=d,status="download failed")
        try:
            B=Prep(f,maxdur=240,fps=2.0 if dur>120 else 3.0,maxframes=480); r=compare(A,B)
        except Exception as e: return dict(url=u,date=d,status="match error "+str(e)[:50])
        png=None
        if verdict(r): png=evidence_png(A,B,r,f"evidence/search_{case}_{abs(hash(u))%10**6}.png",f"{case} cand {u[-40:]} date={d} consistent={r['consistent']} inl={r['max_inliers']}")
        os.remove(f)
        return dict(url=u,date=d,status="checked",match=verdict(r),consistent=r["consistent"],max_inliers=r["max_inliers"],png=png,is_truth=any(t in u for t in truth))
    items=list(cands.items())[:16]
    with cf.ThreadPoolExecutor(4) as ex:
        res=[x for x in ex.map(work,items) if x]
    hits=[x for x in res if x.get("match")]
    out=dict(case=case,query=key,qdate=qdate,plan=plan,n_cands=len(cands),results=res,
             hit_true=any(h["is_truth"] for h in hits),hit_earlier=any(h["date"]<qd for h in hits if h["date"] not in("NA","None")) or any(h["is_truth"] for h in hits),
             hits=hits,cost_muse=c_or,tavily_credits=cost["tav"]-tav0,sec=round(time.time()-t0))
    json.dump(out,open(f"bench/search_{case}.json","w"),indent=1,default=str)
    print(case,"hit_true",out["hit_true"],"hit_earlier",out["hit_earlier"],"cands",len(cands),"hits",[(h["url"],h["date"],h["consistent"],h["max_inliers"]) for h in hits],"muse$",round(c_or,4),"tav",out["tavily_credits"],out["sec"],"s",flush=True)
if __name__=="__main__":
    for c in sys.argv[1:]: run(c)
