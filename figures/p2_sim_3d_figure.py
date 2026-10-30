# Usage: python figures/p2_sim_3d_figure.py data/picai_labels/csPCa_lesion_delineations/human_expert/resampled
# Panel F values are taken from results/lesion_sim_3d_N30.txt (6 mm offset, 0.4 mm noise).
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, pickle, lesion_sim_3d as S, matplotlib
from sklearn.cluster import KMeans
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from skimage import measure
if os.path.exists('Ls3d.pkl'): Ls=pickle.load(open('Ls3d.pkl','rb'))
else:
    Ls=S.load(sys.argv[1])
    for L in Ls:
        L['n']=S.n_devices(L['D']); L['slots']=KMeans(L['n'],n_init=10,random_state=0).fit(L['xyz']).cluster_centers_
    pickle.dump(Ls,open('Ls3d.pkl','wb'))
def run(L,seed,mag,noise,ad):
    # replicate simulate but also return tips
    rng=np.random.default_rng(seed); n=L['n']
    ang=rng.uniform(0,2*np.pi); m=rng.uniform(0.5,1.0)*mag; a,c=m*np.cos(ang),m*np.sin(ang); b,d=rng.uniform(-.18,.18,2)
    zb=rng.uniform(-1,1)*2.0; nz=np.c_[rng.normal(0,noise,(n,2)),rng.normal(0,1.0,n)]
    l,sf,sn=10.0,2.5,max(noise,.1); kf=lambda A,B: sf**2*np.exp(-0.5*((A[:,None]-B[None])**2).sum(-1)/l**2)
    X=np.empty((0,3));Y=np.empty((0,3));tips=[];hits=[]
    for k,slot in enumerate(L['slots']):
        tp=slot+np.array([0,0,S.DEV_LEN/2]); aim=tp.copy()
        if ad and len(X): aim=tp-(kf(tp[None],X)@np.linalg.solve(kf(X,X)+sn**2*np.eye(len(X)),Y))[0]
        err=np.array([a+b*aim[1],c+d*aim[0],zb])+nz[k]; tip=aim+err
        lv=tip[None]-np.c_[np.zeros((7,2)),S.LEVELS]; h=S.inside(L,lv); tips.append(tip); hits.append(h)
        X=np.vstack([X,aim]);Y=np.vstack([Y,err])
    return np.array(tips),np.array(hits)
pick=None
for i,L in enumerate(Ls):
    if not (11<=L['D']<=14 and L['n']==3): continue
    for s in range(30):
        ts,hs=run(L,1000*i+s,6,0.4,False); ta,ha=run(L,1000*i+s,6,0.4,True)
        if hs[1:].mean()<0.25 and ha[1:].mean()>0.8 and hs[0].mean()>0.3:
            pick=(i,s,L,ts,hs,ta,ha); break
    if pick: break
i,s,L,ts,hs,ta,ha=pick; print(L['case'],L['D'],s,hs.mean(1),ha.mean(1))
plt.rcParams.update({'font.size':6,'font.family':'DejaVu Sans','axes.linewidth':0.6})
fig=plt.figure(figsize=(2.7,1.45))
ax=fig.add_axes([0.0,0.22,0.36,0.74])
m=L['mask']; sp=L['sp']; ctr=L['ctr']
dv=(ts[1:,:2]-ta[1:,:2]).mean(0); u=dv/np.linalg.norm(dv)
P=lambda q: np.atleast_2d(q)[:,:2]@u
# project lesion voxels onto (u, z) plane and outline the occupied region
ijk=np.argwhere(m); xyz=ijk*sp-ctr; hu=xyz[:,:2]@u; hz=xyz[:,2]
res=0.25; gx=np.arange(hu.min()-2,hu.max()+2,res); gz=np.arange(hz.min()-3,hz.max()+3,res)
G=np.zeros((len(gx),len(gz)))
for a_,z_ in zip(hu,hz):
    i0=int((a_-gx[0])/res); j0=int((z_-sp[2]/2-gz[0])/res); j1=int((z_+sp[2]/2-gz[0])/res)
    G[max(i0-1,0):i0+2,max(j0,0):j1+1]=1
for c in measure.find_contours(G,0.5):
    ax.plot(gx[0]+c[:,0]*res,gz[0]+c[:,1]*res,color='#b2182b',lw=1)
for k,sl in enumerate(L['slots']): ax.plot(P(sl)[0],sl[2],'+',color='k',ms=5,mew=0.9)
def draw(t,h,col,dx):
    zs=t[2]-S.LEVELS; t=t.copy(); t[0]=P(t)[0]
    ax.plot([t[0]+dx]*2,[t[2],t[2]-S.DEV_LEN],color=col,lw=1.6,solid_capstyle='butt',alpha=0.55)
    ax.scatter([t[0]+dx]*7,zs,s=4,c=[col if q else 'white' for q in h],edgecolors=col,linewidths=0.4,zorder=3)
for k in range(L['n']):
    if k==0: draw(ts[0],hs[0],'#999999',0)
    else: draw(ts[k],hs[k],'#555555',0); draw(ta[k],ha[k],'#2166ac',0)
allx=np.r_[P(ts),P(ta)]; allz=np.r_[ts[:,2],ta[:,2],ts[:,2]-6.5,ta[:,2]-6.5]
lim=max(np.abs(allx).max(),np.abs(allz).max(),L['D']/2)+1.2
ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal'); ax.axis('off')
ax.plot([-lim+0.3,-lim+5.3],[-lim+0.5]*2,'k-',lw=1); ax.text(-lim+2.8,-lim+0.9,'5 mm',ha='center',va='bottom',fontsize=5.5)
ax.annotate('',xy=(lim-0.4,lim-0.5),xytext=(lim-0.4,lim-4.5),arrowprops=dict(arrowstyle='->',lw=0.6))
ax.text(lim-0.9,lim-2.5,'needle',rotation=90,ha='right',va='center',fontsize=4.8)
ax.text(-lim,lim,'E',fontweight='bold',fontsize=8,va='top')
from matplotlib.lines import Line2D
hd=[Line2D([],[],marker='+',ls='',color='k',ms=4,label='plan'),Line2D([],[],color='#555555',lw=1.6,alpha=.55,label='static'),
    Line2D([],[],color='#2166ac',lw=1.6,alpha=.55,label='adaptive'),Line2D([],[],color='#999999',lw=1.6,alpha=.55,label='device 1')]
ax.legend(handles=hd,loc='upper center',bbox_to_anchor=(0.5,0.0),ncol=2,fontsize=5,frameon=False,handlelength=1.2,columnspacing=0.6,labelspacing=0.2,borderaxespad=0)
# F
lab=['$\\leq$10','10–15','15–20','20–27','>27']
st=[9.7,31.3,46.1,57.6,81.7]; ad=[37.2,63.7,74.5,81.3,90.7]; ce=[72.7,87.3,93.6,97.1,98.3]
bx=fig.add_axes([0.50,0.25,0.49,0.62]); x=np.arange(5)
bx.bar(x-0.19,st,0.36,color='#999999',label='static'); bx.bar(x+0.19,ad,0.36,color='#2166ac',label='adaptive')
for xi,c in zip(x,ce): bx.plot([xi-0.4,xi+0.4],[c,c],'k-',lw=0.8)
bx.plot([],[],'k-',lw=0.8,label='no error')
bx.set_xticks(x); bx.set_xticklabels(lab,fontsize=5.2); bx.set_xlabel('largest lesion dimension (mm)',labelpad=1)
bx.set_ylabel('reservoirs in lesion (%)',labelpad=1); bx.set_ylim(0,100); bx.tick_params(length=2,pad=1)
bx.spines[['top','right']].set_visible(False)
bx.legend(fontsize=4.8,frameon=False,loc='lower left',bbox_to_anchor=(0.0,1.0),ncol=3,handlelength=1.0,columnspacing=0.5,borderaxespad=0.1)
bx.text(-0.32,1.16,'F',transform=bx.transAxes,fontweight='bold',fontsize=8,va='top')
fig.savefig('p2_sim.pdf'); fig.savefig('p2_sim.png',dpi=300)
