import cv2,glob,numpy as np,sys
K={"S1":"x_iambmane","S2":"x_dmuppavarapu","S3":"yts_viral_nalagarh","S5":"ig_viral_italy","S6":"yts_sainj","S7":"yt_wayanad_el","S8":"yt_hum","S9":"yt_tauktae2"}
for c,k in K.items():
    cap=cv2.VideoCapture(glob.glob(f"data/videos/{k}.*")[0]); n=int(cap.get(7)); fr=[]
    for i in np.linspace(n*0.05,n*0.95,12).astype(int):
        cap.set(1,i); ok,f=cap.read()
        if ok: fr.append((cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var(),i,f))
    fr.sort(key=lambda x:-x[0])
    # pick top sharp but spread
    pick=[];
    for s,i,f in fr:
        if all(abs(i-j)>n*0.15 for _,j,_ in pick): pick.append((s,i,f))
        if len(pick)==3: break
    for m,(s,i,f) in enumerate(pick): cv2.imwrite(f".cache/rev/{c}_{m}.jpg",f)
    h,w=pick[0][2].shape[:2]; print(c,w,h,[p[1] for p in pick])
    mont=np.hstack([cv2.resize(p[2],(240,int(240*h/w))) for p in pick]); cv2.imwrite(f".cache/rev/{c}_m.jpg",mont)
