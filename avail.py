import json,subprocess,sys,concurrent.futures as cf,os
cases=[json.loads(l) for l in open("bench/cases.jsonl")]
urls=set()
for c in cases:
    for k in("viral_urls","original_urls"):
        for u in c[k]:
            if any(h in u for h in("youtube","instagram","facebook","x.com","newsflare")): urls.add(u)
def chk(u):
    r=subprocess.run(["uv","run","yt-dlp","--simulate","--no-warnings","-q","--print","%(duration)s|%(upload_date)s|%(uploader)s",u],capture_output=True,text=True,timeout=120)
    return u,r.returncode==0,(r.stdout.strip() or r.stderr.strip().splitlines()[-1][:120] if (r.stdout or r.stderr) else "")
res={}
with cf.ThreadPoolExecutor(6) as ex:
    for u,ok,m in ex.map(chk,sorted(urls)):
        res[u]=(ok,m); print(ok,u,m[:100])
json.dump(res,open("bench/availability.json","w"),indent=1)
