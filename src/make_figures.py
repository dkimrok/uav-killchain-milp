#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_figures.py — regenerate Figures 1-4 (grayscale, 300 dpi) into ../figures.

Figures 3 and 4 are computed from results/expI_raw.csv and results/expII_raw.csv.
Figure 1 is drawn from a live instance (N_P = 6, seed 42).
Figure 2 is a schematic of the coupling constraints and uses illustrative times.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
from killchain_auto import generate_dataframes, ATK_DEPOTS
import joint_model as J

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures')
os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,
                     'axes.linewidth':0.8,'savefig.dpi':300,'savefig.bbox':'tight'})
GY='0.45'

# ---------------- Figure III-1: four-zone layout -------------------------------
dp,dt,_,_ = generate_dataframes(J.cfg(300,3,6,42))
fig,ax = plt.subplots(figsize=(4.6,4.6))
for x0 in (0,50):
    for y0 in (0,50):
        ax.add_patch(Rectangle((x0,y0),50,50,fill=False,ec=GY,lw=0.8,ls='--'))
pz={r.point_id:int(r.zone) for r in dp.itertuples(index=False)}
px={r.point_id:(r.x,r.y) for r in dp.itertuples(index=False)}
for t in dt.itertuples(index=False):
    x,y=px[t.point_id]
    ax.plot([x,t.x],[y,t.y],'-',color='0.75',lw=0.6,zorder=1)
ax.scatter([p[0] for p in px.values()],[p[1] for p in px.values()],s=16,marker='o',
           facecolors='white',edgecolors='k',lw=0.8,zorder=3,label='Reconnaissance point $p$')
ax.scatter(dt.x,dt.y,s=20,marker='x',c='k',lw=0.9,zorder=3,label='Target $k$')
ax.scatter([0],[0],s=290,marker='s',facecolors='white',edgecolors='k',lw=1.3,zorder=4)
for z,(dx,dy) in ATK_DEPOTS.items():
    ax.scatter([dx],[dy],s=95,marker='^',c='k',zorder=6)
    off = {(0,0):(11,6),(100,0):(-20,6),(0,100):(11,-16),(100,100):(-20,-16)}[(dx,dy)]
    ax.annotate(f'$0_{z}$',(dx,dy),textcoords='offset points',xytext=off,fontsize=9)
ax.annotate('$0_R$',(0,0),textcoords='offset points',xytext=(-6,-20),fontsize=9,ha='center')
for z,(cx,cy) in {1:(25,25),2:(75,25),3:(25,75),4:(75,75)}.items():
    ax.text(cx,cy,f'BMOA {z}',ha='center',va='center',fontsize=9,color=GY,alpha=.85)
ax.scatter([],[],s=95,marker='^',c='k',label='Attack holding point $0_a$')
ax.scatter([],[],s=140,marker='s',facecolors='white',edgecolors='k',label='Reconnaissance base $0_R$ (co-located with $0_1$)')
ax.set_xlim(-11,111); ax.set_ylim(-13,111); ax.set_aspect('equal')
ax.set_xlabel('km'); ax.set_ylabel('km')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.14),ncol=2,frameon=False,fontsize=8,
          handletextpad=0.4,columnspacing=1.2)
for s in ('top','right'): ax.spines[s].set_visible(False)
fig.savefig(os.path.join(OUT,'Figure_1.png')); plt.close(fig)

# ---------------- Figure III-2: endogenous window timeline ---------------------
fig,ax = plt.subplots(figsize=(6.4,3.0))
D=[6.0,13.5,21.0,29.0]; DK=[5.0,5.0,4.0,5.0]
T=[8.2,14.6,21.8,30.2]; names=['$k_1$','$k_2$','$k_3$','$k_4$']
yR=3.15
ax.hlines(yR,0,34,color='k',lw=1.0)
for i,d in enumerate(D):
    ax.plot(d,yR,'o',ms=5,mfc='white',mec='k',mew=1.0,zorder=4)
    ax.annotate(f'$D_{{p_{i+1}}}$',(d,yR),textcoords='offset points',xytext=(-8,9),fontsize=8.5)
ax.text(-0.6,yR,'Reconnaissance\nUAV',ha='right',va='center',fontsize=8.5)
for i,(d,dk,t) in enumerate(zip(D,DK,T)):
    y=2.25-i*0.62
    ax.add_patch(Rectangle((d,y-0.13),dk,0.26,facecolor='0.86',edgecolor='0.35',lw=0.7))
    ax.annotate('',xy=(d,y-0.30),xytext=(d+dk,y-0.30),
                arrowprops=dict(arrowstyle='<->',lw=0.6,color='0.35',shrinkA=0,shrinkB=0))
    ax.text(d+dk/2,y-0.52,f'$\\Delta_{{{i+1}}}$',ha='center',fontsize=8,color='0.3')
    ax.plot(t,y,'v',ms=7,c='k',zorder=5)
    ax.annotate(f'$T_{{a,{i+1}}}$',(t,y),textcoords='offset points',xytext=(4,5),fontsize=8.5)
    ax.text(d-0.5,y,names[i],ha='right',va='center',fontsize=9)
    ax.vlines(d,y,yR,color='0.7',lw=0.6,ls=':')
    if i>0:
        ax.annotate('',xy=(t,y+0.13),xytext=(T[i-1],2.25-(i-1)*0.62-0.13),
                    arrowprops=dict(arrowstyle='->',lw=0.8,color='k',
                                    connectionstyle='arc3,rad=-0.18'))
ax.annotate('launch on signal',xy=(D[0],yR),xytext=(D[0]-4.6,1.15),fontsize=8,
            arrowprops=dict(arrowstyle='->',lw=0.7,color='0.3'))
ax.text(24.6,2.62,'$s + d_{ij}/v_A$',fontsize=8.5,color='0.2')
ax.set_xlim(-0.5,34.5); ax.set_ylim(-0.55,3.75)
ax.set_xlabel('time (min)'); ax.set_yticks([])
for s in ('top','right','left'): ax.spines[s].set_visible(False)
fig.savefig(os.path.join(OUT,'Figure_2.png')); plt.close(fig)

# ---------------- Figure IV-1: interaction plots -------------------------------
fig,axes = plt.subplots(1,2,figsize=(6.6,2.9),sharey=True)
import pandas as pd
RES=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results')
raw={e:pd.read_csv(os.path.join(RES,f'exp{e}_raw.csv')) for e in ('I','II')}
def cellmean(e,v,dk):
    d=raw[e]; lv=sorted(d.N_P.unique())
    d=d[(d.v_A==v)&(d.dk_min==dk)&(d.N_P.isin([lv[0],lv[2]]))]
    return d.hr.mean()
data={f'Experiment {e}':{dk:[cellmean(e,300,dk),cellmean(e,1110,dk)] for dk in (3,9)}
      for e in ('I','II')}
for ax,(title,d) in zip(axes,data.items()):
    for dk,mk,ls,lab in [(3,'s','--','$\\Delta k_{min}$ = 3 min'),(9,'o','-','$\\Delta k_{min}$ = 9 min')]:
        ax.plot([0,1],d[dk],ls,marker=mk,color='k',ms=6,lw=1.1,mfc='white' if dk==3 else 'k',label=lab)
    ax.set_xticks([0,1]); ax.set_xticklabels(['300','1,110'])
    ax.set_xlabel('$v_A$ (km/h)'); ax.set_title(title,fontsize=9.5)
    ax.set_xlim(-0.25,1.25); ax.set_ylim(0.30,1.06)
    ax.grid(axis='y',color='0.9',lw=0.6); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
axes[0].set_ylabel('Hit Ratio')
axes[1].legend(frameon=False,fontsize=8,loc='lower right')
fig.savefig(os.path.join(OUT,'Figure_3.png')); plt.close(fig)

# ---------------- Figure IV-2: structural ceiling ------------------------------
fig,ax=plt.subplots(figsize=(4.9,3.2))
r=np.linspace(0.6,2.2,400)
ax.plot(r,np.minimum(1.0,r),'-',color='0.35',lw=1.2,label='Ceiling  $\\min(1,\\,W/N_P)$')
allr=pd.concat([raw['I'],raw['II']],ignore_index=True)
fac=allr[allr.ptype=='factorial']
ceil_x=[];ceil_y=[];low_x=[];low_y=[]
for NP in sorted(fac.N_P.unique()):
    g=fac[fac.N_P==NP]
    ceil_x.append(8.0/NP); ceil_y.append(g[~((g.v_A==300)&(g.dk_min==3))].hr.mean())
    low_x.append(8.0/NP);  low_y.append(g[(g.v_A==300)&(g.dk_min==3)].hr.mean())
ax.scatter(ceil_x,ceil_y,s=46,marker='o',c='k',zorder=5,label='Corners with $v_A$ or $\\Delta k$ high')
ax.scatter(low_x,low_y,s=52,marker='v',facecolors='white',edgecolors='k',lw=1.1,zorder=5,
           label='Low-speed narrow-window corner')
for x,y in zip(low_x,low_y):
    ax.vlines(x,y,min(1.0,x),color='0.7',lw=0.7,ls=':')
for x,lab in zip(ceil_x,['$N_P$=4','6','8','10']):
    ax.annotate(lab,(x,1.028 if x>=1 else 0.828),ha='center',fontsize=8,color='0.3')
ax.axvline(1.0,color='0.6',lw=0.8,ls='-.')
ax.text(1.42,0.285,'munition surplus',fontsize=8,color='0.3',ha='center')
ax.text(0.87,0.285,'munition binding',fontsize=8,color='0.3',ha='center')
ax.set_xlabel('Weapon-to-target ratio  $W/N_P$'); ax.set_ylabel('Hit Ratio')
ax.set_xlim(0.62,2.18); ax.set_ylim(0.24,1.10)
ax.invert_xaxis()
ax.legend(frameon=False,fontsize=8,loc='center left')
ax.grid(color='0.92',lw=0.6); ax.set_axisbelow(True)
for s in ('top','right'): ax.spines[s].set_visible(False)
fig.savefig(os.path.join(OUT,'Figure_4.png')); plt.close(fig)
print('figures written')
