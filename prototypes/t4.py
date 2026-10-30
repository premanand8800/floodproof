import json,re,sys
a=json.load(open("bench/articles.json"))
keys=list(a)
lo,hi=int(sys.argv[1]),int(sys.argv[2])
for i,u in enumerate(keys[lo:hi],lo):
    t=a[u]
    print(f"\n### [{i}] {u}")
    # drop image lines
    lines=[l for l in re.split(r"\n+",t) if l.strip() and not l.startswith("![") ]
    out=[]
    for l in lines:
        if re.search(r"(youtube\.com|youtu\.be|instagram\.com|facebook\.com|tiktok|dailymotion|vimeo|twitter\.com|x\.com|archive\.)",l) and re.search(r"originally|original|uploaded|posted|shared|published|channel|found|traced|footage|video",l,re.I):
            l=re.sub(r"!?\[[^\]]*\]\((?!http)[^)]*\)","",l)
            out.append(l[:380])
    print("\n".join(out[:5]))
