import os, json, requests
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
TAV = os.environ["TAVILY_API_KEY"]; OR = os.environ["OPENROUTER_API_KEY"]
def tavily(q, n=8, domains=None, extract=False, **kw):
    body = {"query": q, "max_results": n, "include_raw_content": extract, **kw}
    if domains: body["include_domains"] = domains
    r = requests.post("https://api.tavily.com/search", json=body, headers={"Authorization": f"Bearer {TAV}"}, timeout=60)
    return r.json()
def tav_extract(urls):
    r = requests.post("https://api.tavily.com/extract", json={"urls": urls}, headers={"Authorization": f"Bearer {TAV}"}, timeout=90)
    return r.json()
