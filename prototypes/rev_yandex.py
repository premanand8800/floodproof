import sys,time,json,glob
from playwright.sync_api import sync_playwright
out={}
cases=sys.argv[1:]
with sync_playwright() as p:
    b=p.chromium.launch(headless=True)
    for c in cases:
        for m in range(3):
            img=f".cache/rev/{c}_{m}.jpg"; key=f"{c}_{m}"
            pg=b.new_context(viewport={"width":1280,"height":800},locale="en-US").new_page()
            try:
                pg.goto("https://yandex.com/images/",timeout=40000); time.sleep(2)
                pg.locator("input[type=file]").first.set_input_files(img); time.sleep(7)
                blocked="captcha" in pg.url or "showcaptcha" in pg.url
                pg.screenshot(path=f".cache/rev/yx_{key}_a.png")
                try:
                    pg.get_by_text("Sites",exact=True).first.click(); time.sleep(4)
                except Exception as e: pass
                pg.screenshot(path=f".cache/rev/yx_{key}_b.png")
                links=pg.evaluate("""()=>[...document.querySelectorAll('a[href^=http]')].map(a=>[a.href,(a.innerText||'').trim().slice(0,120)]).filter(x=>!x[0].includes('yandex.'))""")
                out[key]=dict(url=pg.url,blocked=blocked,links=links,text=pg.inner_text("body")[:3000])
            except Exception as e:
                out[key]=dict(err=str(e)[:200])
            print(key,out[key].get("blocked"),len(out[key].get("links",[])),flush=True)
            pg.context.close(); time.sleep(2)
    b.close()
json.dump(out,open(f".cache/rev/yx_{'_'.join(cases)}.json","w"),indent=1,ensure_ascii=False)
