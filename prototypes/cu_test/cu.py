import os, sys, json, time, base64, re, httpx
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
load_dotenv("../.env")
KEY=os.environ["OPENROUTER_API_KEY"]
SYS="""You control a web browser (1280x800 viewport) to accomplish a goal. Each turn you see a screenshot of the current page.
Reply with ONE JSON object only, no prose, choosing one action:
{"action":"goto","url":"https://..."}
{"action":"click","x":int,"y":int}   (x,y normalized 0-1000 across screenshot width/height)
{"action":"type","text":"..."}  (types into focused field; add "enter":true to press Enter after)
{"action":"press","key":"Enter"}
{"action":"scroll","dy":int}  (positive=down)
{"action":"done","answer":"...","evidence":"what on the visible page supports this"}
Only use done with facts you can SEE on the current screenshot. If blocked (captcha/login), say so in done. Add a short "thought" field first."""
def llm(model, goal, hist, shot, url=''):
    msgs=[{"role":"system","content":SYS},{"role":"user","content":[
      {"type":"text","text":f"GOAL: {goal}\nCURRENT PAGE URL: {url}\nACTION HISTORY:\n"+("\n".join(hist[-8:]) or "(none)")+"\nCurrent screenshot:"},
      {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(shot).decode()}}]}]
    r=httpx.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":"Bearer "+KEY},
      json={"model":model,"messages":msgs,"max_tokens":1500,"usage":{"include":True}},timeout=180)
    j=r.json()
    if "choices" not in j: return None,str(j)[:300],0
    txt=j["choices"][0]["message"].get("content") or ""
    return txt,None,(j.get("usage") or {}).get("cost",0)
def parse(t):
    m=re.search(r"\{.*\}",t,re.S)
    return json.loads(m.group(0)) if m else None
def run(name,model,goal,start,maxsteps=20):
    d=f".cache/{name}"; os.makedirs(d,exist_ok=True)
    log=[];hist=[];cost=0;res=None
    with sync_playwright() as p:
        b=p.chromium.launch(headless=True)
        pg=b.new_context(viewport={"width":1280,"height":800},locale="en-US").new_page()
        try: pg.goto(start,timeout=30000)
        except Exception as e: pass
        for i in range(maxsteps):
            time.sleep(1.5)
            shot=pg.screenshot(type="jpeg",quality=55)
            open(f"{d}/{i:02d}.jpg","wb").write(shot)
            t0=time.time(); txt,err,c=llm(model,goal,hist,shot,pg.url); lat=time.time()-t0; cost+=c
            a=None
            try: a=parse(txt) if txt else None
            except Exception: pass
            log.append({"step":i,"url":pg.url,"lat":round(lat,1),"cost":c,"raw":(txt or err)[:600],"action":a})
            if not a: hist.append(f"{i}: invalid reply"); continue
            act=a.get("action")
            try:
                if act=="done": res=a;break
                elif act=="goto": pg.goto(a["url"],timeout=30000)
                elif act=="click": pg.mouse.click(a["x"]*1.28,a["y"]*0.8)
                elif act=="type":
                    pg.keyboard.type(a["text"])
                    if a.get("enter"): pg.keyboard.press("Enter")
                elif act=="press": pg.keyboard.press(a["key"])
                elif act=="scroll": pg.mouse.wheel(0,a["dy"])
                hist.append(f"{i}: {json.dumps({k:v for k,v in a.items() if k!='thought'})}")
            except Exception as e:
                hist.append(f"{i}: {act} failed: {str(e)[:80]}")
        out={"name":name,"model":model,"steps":len(log),"cost":cost,"result":res,"final_url":pg.url,"avg_lat":round(sum(l['lat'] for l in log)/max(1,len(log)),1),"log":log}
        b.close()
    json.dump(out,open(f"{d}/log.json","w"),indent=1)
    print(name,model,"steps",len(log),"cost",round(cost,4),"lat",out["avg_lat"],"\n  RESULT",json.dumps(res)[:500],"\n  url",out["final_url"],flush=True)
T={"t1":("Find the YouTube video of the Dharali (Uttarkashi) flash flood of August 2025 that shows the flood hitting the village; report its URL, channel, and upload date.","https://www.youtube.com"),
"t2":("Given this description of a viral clip: a muddy torrent sweeping away a multi-storey hotel on a river bank in a Himalayan town. Find the earliest upload you can and its date.","https://duckduckgo.com"),
"t3":("Open the page and report the death toll stated in the infobox.","https://en.wikipedia.org/wiki/2024_Nepal_floods")}
if __name__=="__main__":
    model,task,tag=sys.argv[1],sys.argv[2],sys.argv[3]
    g,s=T[task]; run(f"{task}_{tag}",model,g,s)
