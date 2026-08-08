"""
killchain_solver.py
────────────────────────────────────────────────────────────────
Kill Chain MILP 최적화 솔버 (4구역 확장판)
  - killchain_example.xlsx 를 입력으로 받아 MILP 풀기
  - 공격 UAV는 구역(zone)별 1대씩 투입, 담당 구역 내 표적만 타격 가능
  - 결과 콘솔 출력 + 지도/타임라인 PNG 저장

Formulation 수정 이력:
  v2 (4구역): 공격 UAV zone 파라미터 추가, M_OBJ 수식 일치,
              D[p] Big-M 선형화 주석 상세화, u/U bound 명시
"""

import math, sys, os, time
import pandas as pd
import pulp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ═══════════════════════════════════════════════════════════════
# 0.  입력 파일 로드
# ═══════════════════════════════════════════════════════════════
def load_data(xls_path: str):
    sheets = pd.read_excel(xls_path, sheet_name=None)
    required = {"points", "targets", "recon_uav", "attack_uav"}
    missing = required - set(sheets.keys())
    if missing:
        raise ValueError(f"Excel 시트 누락: {missing}")

    points     = sheets["points"]
    targets    = sheets["targets"]
    recon_uav  = sheets["recon_uav"]
    attack_uav = sheets["attack_uav"]

    # zone 컬럼 필수 검증
    if "zone" not in points.columns:
        raise ValueError("points 시트에 'zone' 컬럼이 필요합니다.")
    if "zone" not in attack_uav.columns:
        raise ValueError("attack_uav 시트에 'zone' 컬럼이 필요합니다.")

    # 타입 보정
    for df in [points, targets, recon_uav, attack_uav]:
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype(str).str.strip()

    return points, targets, recon_uav, attack_uav


def euclid(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


# ═══════════════════════════════════════════════════════════════
# 1.  MILP 모델 빌드 + 풀기
# ═══════════════════════════════════════════════════════════════
def build_and_solve(xls_path: str, time_limit: int = 120, msg: bool = True):

    t_total_start = time.time()
    points, targets, recon_uav, attack_uav = load_data(xls_path)

    # ── 집합 ──────────────────────────────────────────────────
    P = points["point_id"].tolist()          # 정찰 지점
    R = recon_uav["uav_id"].tolist()         # 정찰 UAV
    K = targets["target_id"].tolist()        # 전체 표적
    A = attack_uav["uav_id"].tolist()        # 공격 UAV (구역별 1대)

    NP = len(P); NK = len(K)

    # ── 좌표 / 파라미터 ───────────────────────────────────────
    p_xy   = {r.point_id: (float(r.x), float(r.y))  for r in points.itertuples(index=False)}
    p_val  = {r.point_id: float(r.value)             for r in points.itertuples(index=False)}
    p_zone = {r.point_id: int(r.zone)                for r in points.itertuples(index=False)}

    k_xy    = {r.target_id: (float(r.x), float(r.y))  for r in targets.itertuples(index=False)}
    k_val   = {r.target_id: float(r.value)             for r in targets.itertuples(index=False)}
    k_win   = {r.target_id: float(r.valid_minutes)     for r in targets.itertuples(index=False)}
    k_point = {r.target_id: str(r.point_id)            for r in targets.itertuples(index=False)}
    # 표적의 구역: 소속 지점의 구역과 동일
    k_zone  = {k: p_zone[k_point[k]] for k in K}

    r_depot = {r.uav_id: (float(r.start_x), float(r.start_y)) for r in recon_uav.itertuples(index=False)}
    r_speed = {r.uav_id: float(r.speed)    for r in recon_uav.itertuples(index=False)}
    r_fuel  = {r.uav_id: float(r.max_dist) for r in recon_uav.itertuples(index=False)}

    a_depot   = {r.uav_id: (float(r.start_x), float(r.start_y)) for r in attack_uav.itertuples(index=False)}
    a_speed   = {r.uav_id: float(r.speed)    for r in attack_uav.itertuples(index=False)}
    a_fuel    = {r.uav_id: float(r.max_dist) for r in attack_uav.itertuples(index=False)}
    a_weapons = {r.uav_id: int(r.weapons)    for r in attack_uav.itertuples(index=False)}
    a_zone    = {r.uav_id: int(r.zone)       for r in attack_uav.itertuples(index=False)}

    # ── 구역별 표적 집합 (핵심: 공격 UAV a는 K_a 내 표적만 타격) ──
    # K_a[a] = { k ∈ K | zone(k) == a_zone[a] }
    K_a = {a: [k for k in K if k_zone[k] == a_zone[a]] for a in A}

    # ── 거리 / 이동시간 ────────────────────────────────────────
    def rec_depot_id(r): return f"DEPOT_{r}"

    rec_xy = {}; d_rec = {}; t_rec = {}
    for r in R:
        dep = rec_depot_id(r)
        rec_xy[(r, dep)] = r_depot[r]
        for p in P:
            rec_xy[(r, p)] = p_xy[p]
        nodes_r = [dep] + P
        for i in nodes_r:
            for j in nodes_r:
                if i == j: continue
                xi, yi = rec_xy[(r, i)]; xj, yj = rec_xy[(r, j)]
                d_rec[(r,i,j)] = euclid(xi, yi, xj, yj)
                t_rec[(r,i,j)] = d_rec[(r,i,j)] / r_speed[r]

    def atk_depot_id(a): return f"DEPOT_{a}"

    atk_xy = {}; d_atk = {}; t_atk = {}
    for a in A:
        dep = atk_depot_id(a)
        atk_xy[(a, dep)] = a_depot[a]
        Ka = K_a[a]
        for k in Ka:
            atk_xy[(a, k)] = k_xy[k]
        nodes_a = [dep] + Ka
        for i in nodes_a:
            for j in nodes_a:
                if i == j: continue
                xi, yi = atk_xy[(a, i)]; xj, yj = atk_xy[(a, j)]
                d_atk[(a,i,j)] = euclid(xi, yi, xj, yj)
                t_atk[(a,i,j)] = d_atk[(a,i,j)] / a_speed[a]

    # ── Big-M 설정 ─────────────────────────────────────────────
    # M_OBJ: 계층적 목적함수 가중치
    #   정찰 1단위 기여 = M_OBJ × w_p > 전체 타격 가치 합 = Σv_k
    #   ∴ M_OBJ = Σv_k + 1 로 충분 (코드와 수식 일치)
    M_OBJ = int(sum(k_val.values())) + 1

    # M_TIME: 가능한 최대 비행 시간 상한 (Big-M 선형화용)
    max_t_rec = max(r_fuel[r] / r_speed[r] for r in R)
    max_t_atk = max(a_fuel[a] / a_speed[a] for a in A)
    M_TIME = (max_t_rec + max_t_atk) * 10

    # ── MILP 모델 ──────────────────────────────────────────────
    prob = pulp.LpProblem("KillChain_4Zone_MTZ", pulp.LpMaximize)

    # ── 결정변수: 정찰 ─────────────────────────────────────────
    x_rec = {}
    for r in R:
        dep = rec_depot_id(r)
        nodes_r = [dep] + P
        for i in nodes_r:
            for j in nodes_r:
                if i != j:
                    x_rec[(r,i,j)] = pulp.LpVariable(f"xR_{r}_{i}_{j}", 0, 1, cat="Binary")

    z   = {(r,p): pulp.LpVariable(f"z_{r}_{p}",   0,  1, cat="Binary") for r in R for p in P}
    tau = {(r,p): pulp.LpVariable(f"tau_{r}_{p}", 0, None)              for r in R for p in P}
    D   = {p:     pulp.LpVariable(f"D_{p}",        0, None)              for p in P}

    # u_{r,p}: MTZ 정찰 서브투어 제거 보조변수
    #   bound: 0 ≤ u_{r,p} ≤ |P|  (formulation Eq. A'-bound)
    #   방문 노드 순서를 1~|P| 범위에서 인코딩; 미방문이면 0으로 강제
    u   = {(r,p): pulp.LpVariable(f"u_{r}_{p}",   0, NP)                for r in R for p in P}

    # ── 결정변수: 공격 ─────────────────────────────────────────
    # 공격 UAV a는 K_a[a] 내 표적에 대해서만 변수 생성
    y_atk = {}
    for a in A:
        dep = atk_depot_id(a)
        Ka  = K_a[a]
        nodes_a = [dep] + Ka
        for i in nodes_a:
            for j in nodes_a:
                if i != j:
                    y_atk[(a,i,j)] = pulp.LpVariable(f"yA_{a}_{i}_{j}", 0, 1, cat="Binary")

    h = {(a,k): pulp.LpVariable(f"h_{a}_{k}", 0, 1, cat="Binary")
         for a in A for k in K_a[a]}
    T = {(a,k): pulp.LpVariable(f"T_{a}_{k}", 0, None)
         for a in A for k in K_a[a]}

    # U_{a,k}: MTZ 공격 서브투어 제거 보조변수
    #   bound: 0 ≤ U_{a,k} ≤ |K_a|  (구역별 표적 수로 상한 결정)
    NK_a = {a: len(K_a[a]) for a in A}
    U = {(a,k): pulp.LpVariable(f"U_{a}_{k}", 0, NK_a[a])
         for a in A for k in K_a[a]}

    # ── 목적함수 (Eq.1) ────────────────────────────────────────
    # Z = M_OBJ × Σ_{p∈P} w_p · Σ_{r∈R} z_{r,p}
    #           + Σ_{a∈A} Σ_{k∈K_a} v_k · h_{a,k}
    # M_OBJ = Σ_{k∈K} v_k + 1  →  정찰 우선 계층적 목적함수
    prob += (
        M_OBJ * pulp.lpSum(p_val[p] * pulp.lpSum(z[(r,p)] for r in R) for p in P)
        + pulp.lpSum(k_val[k] * h[(a,k)] for a in A for k in K_a[a])
    )

    # ═══════════════════════════════════════════════════════════
    # 제약조건 – 정찰 (Eq. A)
    # ═══════════════════════════════════════════════════════════
    # (A-1) 각 지점은 최대 1대의 정찰 UAV가 방문
    for p in P:
        prob += pulp.lpSum(z[(r,p)] for r in R) <= 1, f"visit_once_{p}"

    for r in R:
        dep = rec_depot_id(r)
        nodes_r = [dep] + P

        # (A-2) depot 출발 ≤ 1, 복귀 ≤ 1
        prob += pulp.lpSum(x_rec[(r,dep,p)] for p in P) <= 1, f"rec_dep_out_{r}"
        prob += pulp.lpSum(x_rec[(r,p,dep)] for p in P) <= 1, f"rec_dep_in_{r}"

        # (A-3) 유입 = 유출 = z (흐름 보존)
        for p in P:
            prob += (pulp.lpSum(x_rec[(r,i,p)] for i in nodes_r if i != p)
                     == z[(r,p)]), f"rec_flow_in_{r}_{p}"
            prob += (pulp.lpSum(x_rec[(r,p,j)] for j in nodes_r if j != p)
                     == z[(r,p)]), f"rec_flow_out_{r}_{p}"

        # (A-4) 연료 제한
        prob += (pulp.lpSum(d_rec[(r,i,j)] * x_rec[(r,i,j)]
                            for i in nodes_r for j in nodes_r if i != j)
                 <= r_fuel[r]), f"rec_fuel_{r}"

        # (A-5) 도착시각 전파 (depot 출발시각 = 0)
        for p in P:
            prob += (tau[(r,p)] >= t_rec[(r,dep,p)] - M_TIME*(1 - x_rec[(r,dep,p)])),\
                    f"tau_depot_{r}_{p}"
        for p in P:
            for q in P:
                if p == q: continue
                prob += (tau[(r,q)] >= tau[(r,p)] + t_rec[(r,p,q)]
                         - M_TIME*(1 - x_rec[(r,p,q)])), f"tau_prop_{r}_{p}_{q}"

    # ── D[p] 정의 (Eq. A-6): 정찰 완료시각 ────────────────────
    # D_p 는 "지점 p 를 방문한 UAV r 의 도착시각 τ_{r,p}" 를 선형적으로 인코딩.
    # 원래 표현: D_p = τ_{r,p} · z_{r,p}  (연속×이진 = 쌍선형)
    # Big-M 선형화 (세 제약):
    #   (i)  D_p ≥ τ_{r,p} − M·(1−z_{r,p})  ∀r  ← z=1 ⟹ D_p ≥ τ_{r,p}
    #   (ii) D_p ≤ τ_{r,p} + M·(1−z_{r,p})  ∀r  ← z=1 ⟹ D_p ≤ τ_{r,p} (등호)
    #   (iii)D_p ≤ M · Σ_r z_{r,p}              ← 미방문이면 D_p = 0
    for p in P:
        prob += D[p] <= M_TIME * pulp.lpSum(z[(r,p)] for r in R), f"D_upper_{p}"  # (iii)
        for r in R:
            prob += D[p] <= tau[(r,p)] + M_TIME*(1 - z[(r,p)]), f"D_le_{r}_{p}"  # (ii)
            prob += D[p] >= tau[(r,p)] - M_TIME*(1 - z[(r,p)]), f"D_ge_{r}_{p}"  # (i)

    # ── MTZ 정찰 서브투어 제거 (Eq. A') ───────────────────────
    # u_{r,p} ∈ [0, |P|]:
    #   방문 노드의 순서값(1~|P|)을 인코딩, 미방문이면 0
    for r in R:
        for p in P:
            prob += u[(r,p)] <= NP * z[(r,p)],  f"mtz_ub_{r}_{p}"   # 미방문 ⟹ u=0
            prob += u[(r,p)] >= z[(r,p)],        f"mtz_lb_{r}_{p}"   # 방문 ⟹ u≥1
        for i in P:
            for j in P:
                if i == j: continue
                prob += (u[(r,i)] - u[(r,j)] + NP*x_rec[(r,i,j)]
                         <= NP - 1), f"mtz_{r}_{i}_{j}"

    # ═══════════════════════════════════════════════════════════
    # 제약조건 – 공격 (Eq. B)
    # ═══════════════════════════════════════════════════════════
    # 각 표적은 최대 1대가 타격 (복수 공격 UAV 운용 시 중복 방지)
    for k in K:
        hitting = [a for a in A if k in K_a[a]]
        if hitting:
            prob += pulp.lpSum(h[(a,k)] for a in hitting) <= 1, f"hit_once_{k}"

    for a in A:
        dep  = atk_depot_id(a)
        Ka   = K_a[a]
        NKa  = NK_a[a]
        nodes_a = [dep] + Ka

        # (B-1) 무장 제한
        prob += (pulp.lpSum(h[(a,k)] for k in Ka) <= a_weapons[a]), f"weapon_{a}"

        # (B-2) depot 출발 ≤ 1, 복귀 ≤ 1
        prob += pulp.lpSum(y_atk[(a,dep,k)] for k in Ka) <= 1, f"atk_dep_out_{a}"
        prob += pulp.lpSum(y_atk[(a,k,dep)] for k in Ka) <= 1, f"atk_dep_in_{a}"

        # (B-3) 유입 = 유출 = h (흐름 보존)
        for k in Ka:
            prob += (pulp.lpSum(y_atk[(a,i,k)] for i in nodes_a if i != k)
                     == h[(a,k)]), f"atk_flow_in_{a}_{k}"
            prob += (pulp.lpSum(y_atk[(a,k,j)] for j in nodes_a if j != k)
                     == h[(a,k)]), f"atk_flow_out_{a}_{k}"

        # (B-4) 연료 제한
        prob += (pulp.lpSum(d_atk[(a,i,j)] * y_atk[(a,i,j)]
                            for i in nodes_a for j in nodes_a if i != j)
                 <= a_fuel[a]), f"atk_fuel_{a}"

        # (B-5) 타격 도착시각 전파 (대기 불가 — 정찰 완료 후 출발)
        # 공격 UAV는 정찰 완료 시각(D[pk])에 출발, 이동 후 도착
        #   depot → k : T[a,k] >= D[pk] + t_atk[dep,k]
        #   i    → j  : T[a,j] >= D[pk_j] + t_atk[i,j]
        # ∴ 타격 가능 조건: t_travel ≤ Δk  (v_A × Δk 교호작용 발현)
        for k in Ka:
            pk = k_point[k]
            prob += (T[(a,k)] >= D[pk] + t_atk[(a,dep,k)]
                     - M_TIME*(1 - y_atk[(a,dep,k)])),\
                    f"T_depot_{a}_{k}"
        for i in Ka:
            for j in Ka:
                if i == j: continue
                pk_j = k_point[j]
                prob += (T[(a,j)] >= D[pk_j] + t_atk[(a,i,j)]
                         - M_TIME*(1 - y_atk[(a,i,j)])), f"T_prop_{a}_{i}_{j}"

        # (B-6) 정찰 선행 조건 + 타임윈도우
        for k in Ka:
            pk = k_point[k]
            prob += h[(a,k)] <= pulp.lpSum(z[(r,pk)] for r in R), f"activate_{a}_{k}"
            prob += T[(a,k)] >= D[pk] - M_TIME*(1 - h[(a,k)]),    f"tw_lo_{a}_{k}"
            prob += T[(a,k)] <= D[pk] + k_win[k] + M_TIME*(1 - h[(a,k)]), f"tw_hi_{a}_{k}"

        # ── MTZ 공격 서브투어 제거 (Eq. B') ───────────────────
        # U_{a,k} ∈ [0, |K_a|]:
        #   구역 내 타격 순서를 인코딩, 미타격이면 0
        for k in Ka:
            prob += U[(a,k)] <= NKa * h[(a,k)], f"atk_mtz_ub_{a}_{k}"  # 미타격 ⟹ U=0
            prob += U[(a,k)] >= h[(a,k)],        f"atk_mtz_lb_{a}_{k}"  # 타격 ⟹ U≥1
        for i in Ka:
            for j in Ka:
                if i == j: continue
                prob += (U[(a,i)] - U[(a,j)] + NKa*y_atk[(a,i,j)]
                         <= NKa - 1), f"atk_mtz_{a}_{i}_{j}"

    # ── 풀기 ───────────────────────────────────────────────────
    #solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    solver = pulp.HiGHS(msg=msg, timeLimit=time_limit, threads=os.cpu_count())
    #solver = pulp.CPLEX_PY(msg=msg, timeLimit=time_limit)
    r'''
    solver = pulp.CPLEX_CMD(
                            path=r"C:\Program Files\IBM\ILOG\CPLEX_Studio1262\cplex\bin\x64_win64\cplex.exe",
                            msg=msg, timeLimit=time_limit)
    '''
    # solver = pulp.GUROBI(msg=msg, timeLimit=time_limit)
    
    start_time = time.time()
    status = prob.solve(solver)
    solver_sec = time.time() - start_time
    total_sec  = time.time() - t_total_start

    # ── 결과 추출 ───────────────────────────────────────────────
    def val(v):
        return pulp.value(v) if pulp.value(v) is not None else 0.0

    explored = {}
    for p in P:
        for r in R:
            if val(z[(r,p)]) > 0.5:
                explored[p] = r

    rec_routes = {}
    for r in R:
        seq = [(p, val(tau[(r,p)])) for p in P if val(z[(r,p)]) > 0.5]
        seq.sort(key=lambda x: x[1])
        rec_routes[r] = seq

    D_vals = {p: val(D[p]) for p in P}

    hits = {}
    for a in A:
        for k in K_a[a]:
            if val(h[(a,k)]) > 0.5:
                hits[k] = a

    atk_routes = {}
    for a in A:
        seq = [(k, val(T[(a,k)])) for k in K_a[a] if val(h[(a,k)]) > 0.5]
        seq.sort(key=lambda x: x[1])
        atk_routes[a] = seq

    T_vals = {(a,k): val(T[(a,k)]) for a in A for k in K_a[a]}

    # 수정 — sol_status로 정확히 구분. OPT/BF 구별용
    sol_st = prob.sol_status
    if sol_st == pulp.constants.LpSolutionOptimal:
        status_str = "Optimal"
    elif sol_st == pulp.constants.LpSolutionIntegerFeasible:
        status_str = "Feasible"   # 시간 초과 BF
    else:
        status_str = pulp.LpStatus[status]

    result = dict(
        status    = status_str,
        obj_value = val(prob.objective),
        M_OBJ     = M_OBJ,
        explored  = explored,
        rec_routes= rec_routes,
        D_vals    = D_vals,
        hits      = hits,
        atk_routes= atk_routes,
        T_vals    = T_vals,
        K_a       = K_a,
        a_zone    = a_zone,
        p_zone    = p_zone,
        k_zone    = k_zone,
        P=P, R=R, K=K, A=A,
        p_xy=p_xy, p_val=p_val,
        k_xy=k_xy, k_val=k_val, k_win=k_win, k_point=k_point,
        r_depot=r_depot, a_depot=a_depot,
        r_speed=r_speed, a_speed=a_speed,
        solver_sec = solver_sec,           # 순수 솔버 시간
        elapsed    = solver_sec,           # 하위 호환용 (기존 코드 유지)
        total_sec  = total_sec,            # 모델 빌드 + 솔버 총 시간
    )
    return result


# ═══════════════════════════════════════════════════════════════
# 7. 결과 출력
# ═══════════════════════════════════════════════════════════════
def print_result(res):
    print("\n" + "="*65)
    print(f"  Status    : {res['status']}")
    print(f"  Solver    : {res['solver_sec']:.2f} sec  (model+solve total: {res['total_sec']:.2f} sec)")
    print(f"  M_OBJ     : {res['M_OBJ']}  (= Σv_k + 1 = {res['M_OBJ']})")
    print(f"  Objective : {res['obj_value']:.2f}")

    rec_val = sum(res['p_val'][p] for p in res['explored'])
    atk_val = sum(res['k_val'][k] for k in res['hits'])
    print(f"  Recon val : {rec_val}   |   Strike val : {atk_val}   "
          f"|   Total : {rec_val+atk_val}")
    print("="*65)

    print("\n[정찰 UAV 경로]")
    for r in res['R']:
        seq = res['rec_routes'][r]
        if not seq:
            print(f"  {r}: 미출격")
            continue
        route = " → ".join(f"{p}(t={t:.1f})" for p,t in seq)
        print(f"  {r}: DEPOT → {route} → DEPOT")

    print("\n[지점 탐색 완료시각]")
    for p in res['P']:
        r  = res['explored'].get(p)
        dp = res['D_vals'][p]
        zn = res['p_zone'][p]
        if r:
            print(f"  {p} [Z{zn}] ({res['p_xy'][p]})  by {r}  D_p={dp:.2f}분  "
                  f"가치={res['p_val'][p]}")
        else:
            print(f"  {p} [Z{zn}] ({res['p_xy'][p]})  미탐색")

    print("\n[공격 UAV 경로 (구역별)]")
    for a in res['A']:
        zn  = res['a_zone'][a]
        seq = res['atk_routes'][a]
        Ka  = res['K_a'][a]
        if not seq:
            print(f"  {a}(Zone{zn}): 미출격  [담당 표적: {Ka}]")
            continue
        route = " → ".join(f"{k}(t={t:.1f})" for k,t in seq)
        print(f"  {a}(Zone{zn}): DEPOT → {route} → DEPOT")

    print("\n[표적 타격 결과]")
    for k in res['K']:
        a  = res['hits'].get(k)
        pk = res['k_point'][k]
        dp = res['D_vals'][pk]
        dlt= res['k_win'][k]
        v  = res['k_val'][k]
        zn = res['k_zone'][k]
        if a:
            tk = res['T_vals'][(a,k)]
            print(f"  {k}[Z{zn}] 가치={v:>4.0f}  소속={pk}  타격 ✓  "
                  f"도착={tk:.1f}분  유효창=[{dp:.1f}, {dp+dlt:.1f}]  by {a}")
        else:
            print(f"  {k}[Z{zn}] 가치={v:>4.0f}  소속={pk}  미타격")
    print("="*65)


# ═══════════════════════════════════════════════════════════════
# 8. 시각화
# ═══════════════════════════════════════════════════════════════
ZONE_COLORS = {1: "#E3F2FD", 2: "#E8F5E9", 3: "#FFF3E0", 4: "#FCE4EC"}
ZONE_EDGE   = {1: "#1565C0", 2: "#2E7D32", 3: "#E65100", 4: "#C62828"}
R_COLORS = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#00695C"]
A_COLORS = {1:"#0D47A1", 2:"#1B5E20", 3:"#BF360C", 4:"#880E4F"}

def draw_map(res, save_path="killchain_map.png"):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor("#F5F5F5")
    ax.set_xlim(-8, 108); ax.set_ylim(-8, 108)
    ax.set_xlabel("X (km)", fontsize=11); ax.set_ylabel("Y (km)", fontsize=11)
    ax.set_title("Kill Chain Operation Map (4-Zone)", fontsize=14,
                 fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.3)

    # 구역 배경
    for zn, (ox, oy) in {1:(0,0), 2:(50,0), 3:(0,50), 4:(50,50)}.items():
        ax.add_patch(plt.Rectangle((ox, oy), 50, 50,
                                   facecolor=ZONE_COLORS[zn],
                                   edgecolor=ZONE_EDGE[zn],
                                   linewidth=1.5, alpha=0.5, zorder=1))
        ax.text(ox+25, oy+25, f"Z{zn}", fontsize=20, color=ZONE_EDGE[zn],
                ha="center", va="center", alpha=0.25, fontweight="bold")

    # depot
    all_depots = set(tuple(v) for v in res['r_depot'].values()) | \
                 set(tuple(v) for v in res['a_depot'].values())
    for d in all_depots:
        ax.plot(*d, "k^", ms=12, zorder=6)
        ax.annotate("DEP", d, xytext=(d[0]+1.5, d[1]+1.5), fontsize=7,
                    fontweight="bold", color="black")

    # 정찰 지점
    for p in res['P']:
        x, y = res['p_xy'][p]; v = res['p_val'][p]; zn = res['p_zone'][p]
        color = ZONE_EDGE[zn] if p in res['explored'] else "#B0BEC5"
        ax.scatter(x, y, s=220+v*60, c=color, zorder=5,
                   edgecolors="navy", linewidths=1.5)
        ax.annotate(f"{p}(w={v})", (x,y), xytext=(x+1.5, y+2),
                    fontsize=8, color="navy", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # 표적
    for k in res['K']:
        x, y = res['k_xy'][k]; v = res['k_val'][k]; dlt = res['k_win'][k]
        hit = k in res['hits']
        c   = "#EF5350" if hit else "#FFCDD2"
        ax.scatter(x, y, s=160+v*10, c=c, marker="*" if hit else "P",
                   zorder=5, edgecolors="darkred" if hit else "gray", linewidths=1.5)
        ax.annotate(f"{k}(v={v},Δ={dlt})", (x,y), xytext=(x+1.5, y-5),
                    fontsize=7, color="darkred" if hit else "#9E9E9E",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    # 정찰 경로
    for ri, r in enumerate(res['R']):
        seq = res['rec_routes'][r]
        if not seq: continue
        col = R_COLORS[ri % len(R_COLORS)]; dep = res['r_depot'][r]
        path = [dep] + [res['p_xy'][p] for p,_ in seq] + [dep]
        for i in range(len(path)-1):
            ax.annotate("", xy=path[i+1], xytext=path[i],
                        arrowprops=dict(arrowstyle="->", color=col, lw=2.0,
                                        alpha=0.8, connectionstyle="arc3,rad=0.05"))
        mx = (path[0][0]+path[1][0])/2; my = (path[0][1]+path[1][1])/2+2
        ax.text(mx, my, r, fontsize=7, color=col, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    # 공격 경로 (구역별 색상)
    for a in res['A']:
        seq = res['atk_routes'][a]
        if not seq: continue
        zn = res['a_zone'][a]; col = A_COLORS[zn]; dep = res['a_depot'][a]
        path = [dep] + [res['k_xy'][k] for k,_ in seq] + [dep]
        for i in range(len(path)-1):
            ax.annotate("", xy=path[i+1], xytext=path[i],
                        arrowprops=dict(arrowstyle="->", color=col, lw=2.3,
                                        linestyle="dashed", alpha=0.9,
                                        connectionstyle="arc3,rad=-0.05"))

    # 범례
    legend_items = []
    for ri, r in enumerate(res['R']):
        if res['rec_routes'][r]:
            legend_items.append(mpatches.Patch(color=R_COLORS[ri%len(R_COLORS)],
                                               label=f"Recon {r}"))
    for a in res['A']:
        if res['atk_routes'][a]:
            zn = res['a_zone'][a]
            legend_items.append(mpatches.Patch(color=A_COLORS[zn],
                                               label=f"Attack {a}(Z{zn})"))
    legend_items += [
        mpatches.Patch(color="#2196F3", label="Explored point"),
        mpatches.Patch(color="#B0BEC5", label="Unexplored point"),
        mpatches.Patch(color="#EF5350", label="Hit target (★)"),
        mpatches.Patch(color="#FFCDD2", label="Missed target"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8,
              framealpha=0.85, ncol=2)

    rec_v = sum(res['p_val'][p] for p in res['explored'])
    atk_v = sum(res['k_val'][k] for k in res['hits'])
    ax.text(105, -6, f"Recon: {rec_v}  |  Strike: {atk_v}  |  Total: {rec_v+atk_v}",
            ha="right", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  지도 저장: {save_path}")


def draw_timeline(res, save_path="killchain_timeline.png"):
    all_uavs = res['R'] + res['A']
    n_rows = len(all_uavs)
    fig, ax = plt.subplots(figsize=(14, 1.5 + n_rows * 0.9))
    ax.set_facecolor("#FAFAFA")

    y_labels = []; max_t = 30

    def draw_leg(ax, y_pos, path_coords, speed, col, label_fn):
        t_cur = 0
        for i in range(len(path_coords)-1):
            dist = math.hypot(path_coords[i+1][0]-path_coords[i][0],
                              path_coords[i+1][1]-path_coords[i][1])
            t_travel = dist / speed
            ax.barh(y_pos, t_travel, left=t_cur, height=0.5,
                    color=col, alpha=0.7, edgecolor="white", linewidth=0.5)
            lbl = label_fn(i)
            if lbl and t_travel > 0.3:
                ax.text(t_cur+t_travel/2, y_pos, lbl,
                        va="center", ha="center", fontsize=7,
                        color="white", fontweight="bold")
            t_cur += t_travel
        return t_cur

    for ri, r in enumerate(res['R']):
        seq = res['rec_routes'][r]; y_pos = n_rows-1-ri; y_labels.append(r)
        if not seq: continue
        col = R_COLORS[ri%len(R_COLORS)]; dep = res['r_depot'][r]
        coords = [dep] + [res['p_xy'][p] for p,_ in seq] + [dep]
        labels = [""] + [f"→{p}" for p,_ in seq] + ["↩"]
        draw_leg(ax, y_pos, coords, res['r_speed'][r], col,
                 lambda i, lbs=labels: lbs[i+1] if i+1<len(lbs) else "")

    for ai, a in enumerate(res['A']):
        seq = res['atk_routes'][a]; y_pos = len(res['A'])-1-ai
        y_labels.append(f"{a}(Z{res['a_zone'][a]})")
        zn = res['a_zone'][a]; col = A_COLORS[zn]
        if not seq: continue
        dep = res['a_depot'][a]
        coords = [dep] + [res['k_xy'][k] for k,_ in seq] + [dep]
        for k, tk in seq:
            pk = res['k_point'][k]; dp = res['D_vals'][pk]; dlt = res['k_win'][k]
            ax.barh(y_pos-0.35, dlt, left=dp, height=0.13,
                    color="#FFC107", alpha=0.9, edgecolor="orange", linewidth=0.5)
            ax.plot(tk, y_pos-0.35, "v", color="darkred", ms=5, zorder=5)
        labels_a = [""] + [f"→{k}" for k,_ in seq] + ["↩"]
        draw_leg(ax, y_pos, coords, res['a_speed'][a], col,
                 lambda i, lbs=labels_a: lbs[i+1] if i+1<len(lbs) else "")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(reversed(y_labels)), fontsize=10)
    ax.set_xlabel("Time (min)", fontsize=11)
    ax.set_title("Kill Chain Operation Timeline (4-Zone)", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, max(max_t, 5))
    ax.grid(axis="x", alpha=0.3)
    patches = [mpatches.Patch(color="#FFC107", alpha=0.9, label="Target time-window"),
               mpatches.Patch(color=R_COLORS[0], alpha=0.7, label="Recon flight"),
               mpatches.Patch(color=list(A_COLORS.values())[0], alpha=0.7, label="Attack flight")]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  타임라인 저장: {save_path}")


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    XLS_PATH = sys.argv[1] if len(sys.argv) > 1 else "killchain_example.xlsx"
    OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nLoading: {XLS_PATH}")
    res = build_and_solve(XLS_PATH, time_limit=60000, msg=False)
    print_result(res)

    map_path  = os.path.join(OUT_DIR, "killchain_map.png")
    time_path = os.path.join(OUT_DIR, "killchain_timeline.png")
    draw_map(res,      save_path=map_path)
    draw_timeline(res, save_path=time_path)
    print("\nDone.")