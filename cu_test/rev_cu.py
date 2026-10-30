import sys,time,json,os
sys.argv=[sys.argv[0]]+sys.argv[1:]
from cu import *
case,img,model=sys.argv[1],sys.argv[2],"meta/muse-spark-open"
goal=("A reverse image search (Yandex) results page is shown for a keyframe of a viral flood/disaster video. Scroll the results (Sites list) and READ them. "
 "Report in done.answer a JSON list of up to 6 results that look like the SAME video footage (YouTube/Instagram/Facebook/X video pages), each with title, channel/site, and any upload date visible. "
 "If none look like video pages of the same scene, answer 'not found'. Do not guess. You have at most 4 scroll steps; you MUST reply with action done by your 5th turn, listing what you have seen so far.")
d=f".cache/rcu_{case}"; os.makedirs(d,exist_ok=True)
hist=[];cost=0;res=None;log=[]
with sync_playwright() as p:
    b=p.chromium.launch(headless=True); pg=b.new_context(viewport={"width":1280,"height":800},locale="en-US").new_page()
    pg.goto("https://yandex.com/images/",timeout=40000); time.sleep(2)
    pg.locator("input[type=file]").first.set_input_files(img); time.sleep(7)
    try: pg.get_by_text("Sites",exact=True).first.click(); time.sleep(3)
    except Exception: pass
    for i in range(6):
        time.sleep(1); shot=pg.screenshot(type="jpeg",quality=55); open(f"{d}/{i:02d}.jpg","wb").write(shot)
        txt,err,c=llm(model,goal,hist,shot,pg.url); cost+=c
        try: a=parse(txt)
        except Exception: a=None
        log.append(((txt or err) or "")[:500])
        if not a: hist.append(f"{i}: invalid"); continue
        if a.get("action")=="done": res=a; break
        try:
            if a["action"]=="scroll": pg.mouse.wheel(0,a.get("dy",600))
            elif a["action"]=="click": pg.mouse.click(a["x"]*1.28,a["y"]*0.8)
            elif a["action"]=="goto": pg.goto(a["url"],timeout=30000)
            hist.append(f"{i}: {json.dumps({k:v for k,v in a.items() if k!='thought'})}")
        except Exception as e: hist.append(f"{i}: failed")
    b.close()
json.dump(dict(cost=cost,res=res,log=log),open(f"{d}/log.json","w"),indent=1)
print(case,"cost",round(cost,4),"steps",len(log),json.dumps(res)[:900])
