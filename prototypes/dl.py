import subprocess,sys,json,os
L={ # key: url
"x_iambmane":"https://x.com/iambmane/status/2092696500640022649",
"ig_leigh":"https://www.instagram.com/reel/DcUFQZhvGzS/",
"x_dmuppavarapu":"https://x.com/dmuppavarapu/status/2092846095751905510",
"yt_leigh":"https://www.youtube.com/watch?v=E5TOaoetGDo",
"yts_viral_nalagarh":"https://www.youtube.com/shorts/Uly-XbejfpA",
"yt_ie_nalagarh":"https://www.youtube.com/watch?v=gG8y--iMl04",
"yt_tribune_nalagarh":"https://www.youtube.com/watch?v=MBLVrGHbJAc",
"ig_viral_italy":"https://www.instagram.com/reel/DM-jOqwztN0/",
"yt_sun_italy":"https://www.youtube.com/watch?v=Nsb9eSzAEgQ",
"ig_sainj":"https://www.instagram.com/p/C_pHwZgvMXg/",
"yts_sainj":"https://www.youtube.com/shorts/k95Yfi__4h0",
"ig_wayanad_el":"https://www.instagram.com/p/C2AMQyYpf_f/",
"yt_wayanad_el":"https://www.youtube.com/watch?v=f7vkQkkrBvY",
"yt_dawn":"https://www.youtube.com/watch?v=GlbjFxMUs6U",
"yt_hum":"https://www.youtube.com/watch?v=OkgI-d_xN_s",
"yt_tauktae1":"https://www.youtube.com/watch?v=bSkLZSRyRu0",
"yt_tauktae2":"https://www.youtube.com/watch?v=ObSniZmyxr8",
"yts_ajesh_buddha":"https://www.youtube.com/shorts/I_AliFdgNl8",
"yts_nepal_vanTV":"https://www.youtube.com/shorts/1oT56HbSnVI",
"ig_dharali_accu":"https://www.instagram.com/reels/DM-rGp0Jcx8/",
"ig_pak2022":"https://www.instagram.com/p/ChzTcCQsZj-/",
"yt_croc":"https://www.youtube.com/watch?v=0mRS62uE19M",
"yt_michael":"https://www.youtube.com/watch?v=kU3Dc6T8Sug",
"yt_japan2015":"https://www.youtube.com/watch?v=EEcotUWktuQ",
"yt_sankei":"https://www.youtube.com/watch?v=5LWNIhMBNYw",
"yt_kompas":"https://www.youtube.com/watch?v=NF35eTdABjM",
"yt_ramban":"https://www.youtube.com/watch?v=0T4UuJWXGB0",
"yt_sikkim_austria":"https://www.youtube.com/watch?v=ObYRYF3d38Y",
"yt_indexp_punjab":"https://www.youtube.com/watch?v=1bI4bXO_E2w",
"yt_manorama_dog":"https://www.youtube.com/watch?v=wbCXA6sWlVo",
"fb_kolkata":"https://www.facebook.com/watch/?v=1841431486754837",
"ig_wayanad_china":"https://www.youtube.com/watch?v=-UBuGHH_pIo",
}
for k,u in L.items():
    out=f"data/videos/{k}.mp4"
    if os.path.exists(out): continue
    r=subprocess.run(["uv","run","yt-dlp","-q","--no-warnings","-f","bv*[height<=480]/b[height<=480]/worst","--download-sections","*0-75","--force-keyframes-at-cuts","--merge-output-format","mp4","--max-filesize","40M","-o",f"data/videos/{k}.%(ext)s",u],capture_output=True,text=True)
    fs=[f for f in os.listdir("data/videos") if f.startswith(k+".")]
    print(k,r.returncode,fs,(r.stderr.strip().splitlines() or [""])[-1][:100] if r.returncode else "")
