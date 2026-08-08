"""joint_model v2 — zone-assigned reconnaissance + honest solver status.

CHANGES vs v1
  1. recon_mode='zone' (default): each BMOA zone is surveyed by one dedicated
     reconnaissance UAV.  D_p stays endogenous (routing/timing optimised), but the
     model separates into 4 small zone subproblems -> tractable.
     recon_mode='free' reproduces v1 (UAVs roam across zones).
  2. HONEST STATUS.  PuLP maps HiGHS kTimeLimit -> LpStatusOptimal, so
     pulp.LpStatus[...] == 'Optimal' does NOT mean optimal.  We read
     HighsModelStatus and the MIP gap directly.
  3. Node-specific Big-M instead of one global constant.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from killchain_auto import generate_dataframes, ATK_DEPOTS
import pulp

V_R = 260/60.0

def cfg(v, dk, NP, s, nR=4, W=8):
    return dict(area_size=100, n_points_per_zone=int(NP), point_value_range=[1,10],
        targets_per_point_range=[1,1], target_value_range=[1,10],
        target_window_range=[int(dk), int(dk+2)], target_offset_range=[2,8],
        n_recon_uav=int(nR), recon_start=[0,0], recon_speed=V_R, recon_max_dist=260*4.6,
        attack_speed=v/60.0, attack_max_dist=1110*1.5, attack_weapons=W, random_seed=int(s))

def _status(prob, tl, elapsed):
    """Return (label, gap). Never trust pulp.LpStatus alone."""
    try:
        import highspy
        sm = prob.solverModel
        ms = sm.getModelStatus()
        gap = float(sm.getInfo().mip_gap)
        name = str(ms).split('.')[-1]
        if ms == highspy.HighsModelStatus.kOptimal:
            return "Optimal", gap
        if ms == highspy.HighsModelStatus.kTimeLimit:
            return ("TimeLimit(feasible)" if gap < 1e30 else "TimeLimit(no incumbent)"), gap
        if ms == highspy.HighsModelStatus.kInfeasible:
            return "Infeasible", float('nan')
        return name, gap
    except Exception:
        lab = pulp.LpStatus[prob.status]
        if elapsed >= 0.98*tl:
            lab = f"{lab}?TimeLimit"
        return lab, float('nan')

def solve(v, dk, NP, seed, nR=4, W=8, F_R=260*4.6, F_A=1110*1.5, s_svc=1.0,
          wmax=2.0, time_limit=900, threads=None, msg=False, recon_mode='zone',
          prune=True, gap=1e-6):
    import time as _t
    dp, dt, _, _ = generate_dataframes(cfg(v, dk, NP, seed, nR, W))
    sp = v/60.0
    pxy={r.point_id:(float(r.x),float(r.y)) for r in dp.itertuples(index=False)}
    pz ={r.point_id:int(r.zone) for r in dp.itertuples(index=False)}
    tg =[(r.target_id,float(r.x),float(r.y),float(r.value),float(r.valid_minutes),
          pz[r.point_id],r.point_id) for r in dt.itertuples(index=False)]
    eps = 1.0/(len(tg)+1)
    zones = sorted(ATK_DEPOTS)
    groups = [[z] for z in zones] if recon_mode=='zone' else [zones]
    tot_hit=0; tot_sv=0.0; labs=[]; gaps=[]; t0=_t.time()
    for grp in groups:
        Pg=[p for p in pxy if pz[p] in grp]
        Rg=list(range(1 if recon_mode=='zone' else nR))
        DEP='D0'; nodesR=[DEP]+Pg
        rxy={DEP:(0.0,0.0)}; rxy.update({p:pxy[p] for p in Pg})
        dR={(i,j):math.hypot(rxy[i][0]-rxy[j][0],rxy[i][1]-rxy[j][1])
            for i in nodesR for j in nodesR if i!=j}
        tR={k:val/V_R for k,val in dR.items()}
        Dmax={p: min(F_R/V_R, sum(sorted(tR.values())[-len(Pg):])) for p in Pg}
        M_R=max(Dmax.values())+max(tR.values())
        m=pulp.LpProblem("jz",pulp.LpMaximize)
        x={(r,i,j):pulp.LpVariable(f"x_{r}_{i}_{j}",cat="Binary") for r in Rg for (i,j) in dR}
        z={(r,p):pulp.LpVariable(f"z_{r}_{p}",cat="Binary") for r in Rg for p in Pg}
        tau={(r,p):pulp.LpVariable(f"ta_{r}_{p}",0,M_R) for r in Rg for p in Pg}
        D={p:pulp.LpVariable(f"D_{p}",0,Dmax[p]) for p in Pg}
        u={(r,p):pulp.LpVariable(f"u_{r}_{p}",0,len(Pg)) for r in Rg for p in Pg}
        for p in Pg: m += pulp.lpSum(z[(r,p)] for r in Rg)==1
        for r in Rg[:-1]:
            m += pulp.lpSum(z[(r,p)] for p in Pg) >= pulp.lpSum(z[(r+1,p)] for p in Pg)
        for r in Rg:
            m += pulp.lpSum(x[(r,DEP,p)] for p in Pg)<=1
            m += pulp.lpSum(x[(r,p,DEP)] for p in Pg)<=1
            m += pulp.lpSum(dR[(i,j)]*x[(r,i,j)] for (i,j) in dR)<=F_R
            for p in Pg:
                m += pulp.lpSum(x[(r,i,p)] for i in nodesR if i!=p)==z[(r,p)]
                m += pulp.lpSum(x[(r,p,j)] for j in nodesR if j!=p)==z[(r,p)]
                m += u[(r,p)]<=len(Pg)*z[(r,p)]; m += u[(r,p)]>=z[(r,p)]
                m += tau[(r,p)]>=tR[(DEP,p)]-M_R*(1-x[(r,DEP,p)])
                m += tau[(r,p)]<=M_R*z[(r,p)]
                m += D[p]>=tau[(r,p)]-M_R*(1-z[(r,p)])
                m += D[p]<=tau[(r,p)]+M_R*(1-z[(r,p)])
            for i in Pg:
                for j in Pg:
                    if i==j: continue
                    m += tau[(r,j)]>=tau[(r,i)]+tR[(i,j)]-M_R*(1-x[(r,i,j)])
                    m += u[(r,i)]-u[(r,j)]+len(Pg)*x[(r,i,j)]<=len(Pg)-1
        obj=[]; hv={}
        for zn in grp:
            Ka=[t for t in tg if t[5]==zn]
            ids=[t[0] for t in Ka]
            pos={'A':ATK_DEPOTS[zn]}; pos.update({t[0]:(t[1],t[2]) for t in Ka})
            val={t[0]:t[3] for t in Ka}; win={t[0]:t[4] for t in Ka}; pk={t[0]:t[6] for t in Ka}
            nd=['A']+ids; n=len(ids)
            dA={(i,j):math.hypot(pos[i][0]-pos[j][0],pos[i][1]-pos[j][1])
                for i in nd for j in nd if i!=j}
            tA={k:val_/sp for k,val_ in dA.items()}
            Tub={k: Dmax[pk[k]]+win[k] for k in ids}
            y={k:pulp.LpVariable(f"y{zn}_{k[0]}_{k[1]}",cat="Binary") for k in dA}
            h={k:pulp.LpVariable(f"h{zn}_{k}",cat="Binary") for k in ids}
            T={k:pulp.LpVariable(f"T{zn}_{k}",0,Tub[k]) for k in ids}
            U={k:pulp.LpVariable(f"U{zn}_{k}",0,n) for k in ids}
            hv.update(h)
            m += pulp.lpSum(h[k] for k in ids)<=W
            m += pulp.lpSum(y[('A',k)] for k in ids)<=1
            m += pulp.lpSum(y[(k,'A')] for k in ids)<=1
            m += pulp.lpSum(dA[k]*y[k] for k in dA)<=F_A
            for k in ids:
                if prune and tA[('A',k)]>win[k]+1e-9: m += y[('A',k)]==0
                m += pulp.lpSum(y[(i,k)] for i in nd if i!=k)==h[k]
                m += pulp.lpSum(y[(k,j)] for j in nd if j!=k)==h[k]
                m += U[k]<=n*h[k]; m += U[k]>=h[k]
                m += T[k]<=Tub[k]*h[k]
                m += T[k]>=D[pk[k]]-Tub[k]*(1-h[k])
                m += T[k]<=D[pk[k]]+win[k]+Tub[k]*(1-h[k])
                m += T[k]>=D[pk[k]]+tA[('A',k)]-(Dmax[pk[k]]+tA[('A',k)])*(1-y[('A',k)])
            for i in ids:
                for j in ids:
                    if i==j: continue
                    Mij=Tub[i]+s_svc+tA[(i,j)]
                    m += T[j]>=T[i]+s_svc+tA[(i,j)]-Mij*(1-y[(i,j)])
                    m += T[j]<=T[i]+s_svc+tA[(i,j)]+wmax+(Tub[j]+Mij)*(1-y[(i,j)])
                    m += U[i]-U[j]+n*y[(i,j)]<=n-1
            obj+=[val[k]*h[k]+eps*h[k] for k in ids]
        m += pulp.lpSum(obj)
        kw=dict(msg=msg,timeLimit=time_limit,gapRel=gap,gapAbs=0.0)
        if threads: kw['threads']=threads
        a=_t.time(); m.solve(pulp.HiGHS(**kw)); el=_t.time()-a
        lab,g=_status(m,time_limit,el); labs.append(lab); gaps.append(g)
        hit=[k for k,vv in hv.items() if vv.value() and vv.value()>0.5]
        kv={t[0]:t[3] for t in tg}
        tot_hit+=len(hit); tot_sv+=sum(kv[k] for k in hit)
    ok=all(l=="Optimal" for l in labs)
    return dict(status="Optimal" if ok else "/".join(sorted(set(labs))),
                proven=ok, max_gap=max([g for g in gaps if g==g], default=float('nan')),
                sv=tot_sv, hr=tot_hit/len(tg), n_hit=tot_hit, n_K=len(tg),
                sec=round(_t.time()-t0,1))
