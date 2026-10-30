import sys,time
from playwright.sync_api import sync_playwright
img=sys.argv[1]; eng=sys.argv[2]
with sync_playwright() as p:
    b=p.chromium.launch(headless=True)
    pg=b.new_context(viewport={"width":1280,"height":800},locale="en-US").new_page()
    url={"bing":"https://www.bing.com/visualsearch","yandex":"https://yandex.com/images/","google":"https://lens.google.com/"}[eng]
    pg.goto(url,timeout=40000); time.sleep(3)
    pg.screenshot(path=f".cache/rev/probe_{eng}_0.png"); print(pg.url)
    fi=pg.locator("input[type=file]"); print("file inputs",fi.count())
    if fi.count()==0 and eng=="yandex":
        pg.locator("[aria-label*='mage search'], .HeaderDesktopActions-CbirButton, button[class*=Cbir]").first.click(); time.sleep(2)
        fi=pg.locator("input[type=file]"); print("file inputs2",fi.count())
    if fi.count():
        fi.first.set_input_files(img); time.sleep(8)
    pg.screenshot(path=f".cache/rev/probe_{eng}_1.png"); print(pg.url); print(pg.inner_text("body")[:600])
    b.close()
