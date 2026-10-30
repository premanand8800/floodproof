from fp_common import *
import re
hits=json.load(open("bench/search_hits.json"))
skip=["/list/","/topic","tags/","webqoof/webqoof-recap","opinion","sikkim-flood-victim","busting","ai-generated","AI-gen","cyberpeace.org/resources/fact-check","meta-ai","weekly-wrap","tulsi","/amp/","haryana","bhagwa","iskcon","praying","factcheck.afp.com/AFP","thequint.com/news/webqoof$","deepfake"]
urls=[u for u,t in hits.items() if not any(s in u for s in skip) and not re.search(r"AI-generated|AI images|Conspiracy|fossil|artwork|Trump|photos|Stone|How China",t)]
urls+=["https://www.boomlive.in/fact-check/viral-nepal-flash-flood-videos-claim-fact-check-old-unrelated-videos-32124"]
urls=list(dict.fromkeys(urls))
print(len(urls))
out={}
for i in range(0,len(urls),5):
    r=tav_extract(urls[i:i+5])
    for x in r.get("results",[]): out[x["url"]]=x["raw_content"]
    for f in r.get("failed_results",[]): print("fail",f.get("url"))
json.dump(out,open("bench/articles.json","w"))
print(len(out))
