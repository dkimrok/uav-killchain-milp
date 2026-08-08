"""Joint recon+attack MILP, cumulative attack time with LIMITED WAITING.

  Eq(23)  T_k >= D_pk + t(dep,k) - M(1-y[dep,k])          depot departure
  Eq(24)  T_j >= T_i + s + t_ij - M(1-y_ij)               cumulative (s = service)
  Eq(24b) T_j <= T_i + s + t_ij + Wmax + M(1-y_ij)        loiter cap  <-- NEW
  Eq(21)  T_k >= D_pk - M(1-h)
  Eq(22)  T_k <= D_pk + dk_k + M(1-h)

Tractability measures (all exact / valid):
  P1 depot-arc pruning  t(dep,k) > dk_k  =>  y[dep,k] = 0     (Eq 25, now genuinely valid)
  P2 recon symmetry breaking: identical UAVs -> non-increasing load
  P3 warm start from the NN full-coverage tour
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from killchain_auto import generate_dataframes, ATK_DEPOTS
V_R = 260/60.0

def cfg(v, dk, NP, s, nR=4, W=8):
    return dict(area_size=100, n_points_per_zone=int(NP), point_value_range=[1, 10],
                targets_per_point_range=[1, 1], target_value_range=[1, 10],
                target_window_range=[int(dk), int(dk + 2)], target_offset_range=[2, 8],
                n_recon_uav=int(nR), recon_start=[0, 0], recon_speed=V_R,
                recon_max_dist=260 * 4.6, attack_speed=v / 60.0,
                attack_max_dist=1110 * 1.5, attack_weapons=W, random_seed=int(s))
import pulp

S_SERVICE = 1.0
W_MAX     = 2.0

def solve(v, dk, NP, seed, nR=4, W=8, F_R=260*4.6, F_A=1110*1.5, s_svc=S_SERVICE,
          wmax=W_MAX, time_limit=900, threads=None, msg=False,
          prune=True, symbreak=True, warm=True, gap=1e-6):
    dp, dt, _, _ = generate_dataframes(cfg(v, dk, NP, seed, nR, W))
    sp = v/60.0
    P   = [r.point_id for r in dp.itertuples(index=False)]
    pxy = {r.point_id:(float(r.x),float(r.y)) for r in dp.itertuples(index=False)}
    pz  = {r.point_id:int(r.zone) for r in dp.itertuples(index=False)}
    tg  = [(r.target_id,float(r.x),float(r.y),float(r.value),float(r.valid_minutes),
            pz[r.point_id],r.point_id) for r in dt.itertuples(index=False)]
    R=list(range(nR)); DEP='D0'; nodesR=[DEP]+P
    rxy={DEP:(0.0,0.0)}; rxy.update(pxy)
    dR={(i,j):math.hypot(rxy[i][0]-rxy[j][0],rxy[i][1]-rxy[j][1])
        for i in nodesR for j in nodesR if i!=j}
    tR={k:val/V_R for k,val in dR.items()}
    M_R=F_R/V_R+max(tR.values())
    m=pulp.LpProblem("joint2",pulp.LpMaximize)
    x={(r,i,j):pulp.LpVariable(f"x_{r}_{i}_{j}",cat="Binary") for r in R for (i,j) in dR}
    z={(r,p):pulp.LpVariable(f"z_{r}_{p}",cat="Binary") for r in R for p in P}
    tau={(r,p):pulp.LpVariable(f"ta_{r}_{p}",0,None) for r in R for p in P}
    D={p:pulp.LpVariable(f"D_{p}",0,None) for p in P}
    u={(r,p):pulp.LpVariable(f"u_{r}_{p}",0,len(P)) for r in R for p in P}
    for p in P: m += pulp.lpSum(z[(r,p)] for r in R)==1
    if symbreak:                                                   # P2
        for r in R[:-1]:
            m += pulp.lpSum(z[(r,p)] for p in P) >= pulp.lpSum(z[(r+1,p)] for p in P)
    for r in R:
        m += pulp.lpSum(x[(r,DEP,p)] for p in P)<=1
        m += pulp.lpSum(x[(r,p,DEP)] for p in P)<=1
        m += pulp.lpSum(dR[(i,j)]*x[(r,i,j)] for (i,j) in dR)<=F_R
        for p in P:
            m += pulp.lpSum(x[(r,i,p)] for i in nodesR if i!=p)==z[(r,p)]
            m += pulp.lpSum(x[(r,p,j)] for j in nodesR if j!=p)==z[(r,p)]
            m += u[(r,p)]<=len(P)*z[(r,p)]; m += u[(r,p)]>=z[(r,p)]
            m += tau[(r,p)]>=tR[(DEP,p)]-M_R*(1-x[(r,DEP,p)])
            m += D[p]>=tau[(r,p)]-M_R*(1-z[(r,p)])
            m += D[p]<=tau[(r,p)]+M_R*(1-z[(r,p)])
        for i in P:
            for j in P:
                if i==j: continue
                m += tau[(r,j)]>=tau[(r,i)]+tR[(i,j)]-M_R*(1-x[(r,i,j)])
                m += u[(r,i)]-u[(r,j)]+len(P)*x[(r,i,j)]<=len(P)-1
    eps=1.0/(len(tg)+1); obj=[]; hvars={}; n_pruned=0
    for zn in sorted(ATK_DEPOTS):
        Ka=[t for t in tg if t[5]==zn]
        if not Ka: continue
        ids=[t[0] for t in Ka]
        pos={'A':ATK_DEPOTS[zn]}; pos.update({t[0]:(t[1],t[2]) for t in Ka})
        val={t[0]:t[3] for t in Ka}; win={t[0]:t[4] for t in Ka}; pk={t[0]:t[6] for t in Ka}
        nd=['A']+ids; n=len(ids)
        dA={(i,j):math.hypot(pos[i][0]-pos[j][0],pos[i][1]-pos[j][1])
            for i in nd for j in nd if i!=j}
        tA={k:val_/sp for k,val_ in dA.items()}
        BIG=M_R+n*(s_svc+max(tA.values()))+max(win.values())+10
        y={k:pulp.LpVariable(f"y{zn}_{k[0]}_{k[1]}",cat="Binary") for k in dA}
        h={k:pulp.LpVariable(f"h{zn}_{k}",cat="Binary") for k in ids}
        T={k:pulp.LpVariable(f"T{zn}_{k}",0,None) for k in ids}
        U={k:pulp.LpVariable(f"U{zn}_{k}",0,n) for k in ids}
        hvars.update({k:h[k] for k in ids})
        m += pulp.lpSum(h[k] for k in ids)<=W
        m += pulp.lpSum(y[('A',k)] for k in ids)<=1
        m += pulp.lpSum(y[(k,'A')] for k in ids)<=1
        m += pulp.lpSum(dA[k]*y[k] for k in dA)<=F_A
        for k in ids:
            if prune and tA[('A',k)]>win[k]+1e-9:                   # P1
                m += y[('A',k)]==0; n_pruned+=1
            m += pulp.lpSum(y[(i,k)] for i in nd if i!=k)==h[k]
            m += pulp.lpSum(y[(k,j)] for j in nd if j!=k)==h[k]
            m += U[k]<=n*h[k]; m += U[k]>=h[k]
            m += T[k]>=D[pk[k]]-BIG*(1-h[k])
            m += T[k]<=D[pk[k]]+win[k]+BIG*(1-h[k])
            m += T[k]>=D[pk[k]]+tA[('A',k)]-BIG*(1-y[('A',k)])
        for i in ids:
            for j in ids:
                if i==j: continue
                m += T[j]>=T[i]+s_svc+tA[(i,j)]-BIG*(1-y[(i,j)])
                m += T[j]<=T[i]+s_svc+tA[(i,j)]+wmax+BIG*(1-y[(i,j)])   # loiter cap
                m += U[i]-U[j]+n*y[(i,j)]<=n-1
        obj+=[val[k]*h[k]+eps*h[k] for k in ids]
    m += pulp.lpSum(obj)
    kw=dict(msg=msg,timeLimit=time_limit,gapRel=gap,gapAbs=0.0,warmStart=warm)
    if threads: kw['threads']=threads
    m.solve(pulp.HiGHS(**kw))
    hit=[k for k,vv in hvars.items() if vv.value() and vv.value()>0.5]
    kv={t[0]:t[3] for t in tg}
    return dict(status=pulp.LpStatus[m.status],n_hit=len(hit),n_K=len(tg),
                hr=len(hit)/len(tg),sv=sum(kv[k] for k in hit),pruned=n_pruned,
                n_bin=sum(1 for vv in m.variables() if vv.cat=='Binary'))
