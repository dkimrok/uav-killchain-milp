"""
killchain_doe.py
══════════════════════════════════════════════════════════════════════
Kill Chain MILP 실험계획법 (Design of Experiments) — 4구역 모델 v4
공격 UAV 속도 × 타임윈도우 × 구역당 정찰 지점 수

■ 재설계 배경 (v3 → v4)
    v3 결과: v_A=300+Δk=3 셀에서만 HR 급락 확인 (교호작용 실재)
    문제점: Low-Low 셀 내 시드 간 분산 과대(CV=52%) → 검정력 부족 (p≈0.13)
    해결:   ① 반복 시드 3→6개 (검정력 80% 달성)
            ② N_P_zone 인자 추가 (W/NP 비율이 HR 상한 결정 → 복합 효과)

■ 솔버 조건 (대기 불가, B-5 수정)
    공격 UAV는 정찰 완료 시각(D[pk])에 출발 → T[a,k] ≥ D[pk] + t_travel
    ∴ 타격 조건: t_travel ≤ Δk

■ 구역 내 이동 시간 (작전 공간 100×100 km, 구역 50×50 km)
    v_A=300km/h  → 평균 이동 7.2min / 최대 14.0min
    v_A=600km/h  → 평균 이동 3.6min / 최대  7.0min
    v_A=1110km/h → 평균 이동 1.9min / 최대  3.8min

■ 고정 제원
    v_R = 260 km/h = 4.333 km/min,  F_R = 1,196 km
    F_A = 1,665 km (항속거리 고정)
    W   = 8 발/구역 UAV (작전상 고정)
    T_pp = 1,  n_R = 4 (정찰 UAV, 이전 실험에서 비유의 → 고정)
    4구역 구조, 공격 UAV 4기 (구역별 1기)

■ 실험 설계
    2³ 완전 요인 배치 (8점) + 중심점 3회 = 11점
    × 반복 시드 6개 = 총 66회

■ 인자 수준
    A  v_A_kmh   공격 UAV 속도 (km/h)  Low=300  / Center=600 / High=1110
    B  dk_min    시간창 최솟값 (분)     Low=3    / Center=6   / High=9
                 → Δk ~ Uniform[dk_min, dk_min+2]
    C  N_P_zone  구역당 정찰 지점 수   Low=4    / Center=6   / High=8
                 (W=8 고정, W/NP: Low=2.0 / Center=1.33 / High=1.0)

■ 기대 교호작용
    v_A×Δk   : v_A 낮고 Δk 좁을 때 HR 급락
    v_A×NP   : v_A 낮을수록 원거리 표적(NP↑) 타격 불가 효과 증폭
    Δk×NP    : Δk 좁고 NP 많을수록 무장+시간 이중 제약

■ 반응변수
    sv          총 타격 가치 (4구역 합계)
    hr          Hit Ratio = 타격 표적 수 / 전체 표적 수
    rr          Recon Ratio = 탐색 지점 수 / 전체 지점 수
    n_hit       타격 표적 수
    weapon_bind 무장 바인딩 비율 (타격=W_FIXED인 구역 수 / 4)
    sec         총 실행시간 (초),  solver_sec: 순수 솔버 시간 (초)
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import sys, os, time, math, itertools, warnings, threading
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import scipy.stats as stats
# matplotlib는 결과 플롯 전용 — 없으면 자동 생략 (핵심 실험 파이프라인은 미사용)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from killchain_auto        import generate_dataframes, save_excel
# 최종 솔버(Big-M 타이트닝 + 정찰 고정, BF 비수렴 해소) 사용
from killchain_solver_fix2 import build_and_solve

# ══════════════════════════════════════════════════════════════════
# 0. 전역 상수
# ══════════════════════════════════════════════════════════════════
V_R        = 260  / 60      # 4.333 km/min (정찰 UAV, 고정)
F_R        = 260  * 4.6     # 1,196 km
F_A        = 1110 * 1.5     # 1,665 km (항속거리 고정, v_A 변화와 무관)
W_FIXED    = 8              # 무장 수 (작전상 고정)
T_PP       = 1              # 지점당 표적 수 고정
N_ZONES    = 4
N_R_FIXED  = 4              # 정찰 UAV 수 고정 (이전 실험에서 비유의)
TIME_LIMIT = 3600           # 솔버 시간 제한 (초)
DK_WIDTH   = 2              # Δk 분포 폭: Δk ~ Uniform[dk_min, dk_min+DK_WIDTH]
R_SEEDS    = [42, 99, 314, 7, 21, 137]   # 6시드 (검정력 80% 확보)

# ── 병렬 실행 설정 ────────────────────────────────────────────────
import os as _os
_cpu = _os.cpu_count() or 8
N_WORKERS            = max(1, _cpu // 4)
N_THREADS_PER_SOLVER = 4

# ── 인자 수준 ─────────────────────────────────────────────────────
LEVELS = {
    "v_A_kmh":  [300,  600, 1110],  # Low=300  / Center=600 / High=1110 (km/h)
    "dk_min":   [3,    6,   9   ],  # Low=3    / Center=6   / High=9    (분)
    "N_P_zone": [4,    6,   8   ],  # Low=4    / Center=6   / High=8    (개/구역)
}

FACTOR_INFO = [
    ("v_A_kmh",  "cV", "Attack Speed (v_A, km/h)"),
    ("dk_min",   "cD", "Time Window min (Δ_k, min)"),
    ("N_P_zone", "cP", "Points/Zone (N_P)"),
]
RESP_INFO = [
    ("sv",          "Strike Value"),
    ("hr",          "Hit Ratio"),
    ("rr",          "Recon Ratio"),
    ("weapon_bind", "Weapon Bind Rate"),
]
PA = ["#EF5350", "#1565C0", "#2E7D32", "#FF6F00"]


# ══════════════════════════════════════════════════════════════════
# 1. cfg dict 생성
# ══════════════════════════════════════════════════════════════════
def make_cfg(v_A_kmh: float, dk_min: float, N_P_zone: int, seed: int) -> dict:
    """실험점 파라미터 → generate_dataframes() 용 cfg dict."""
    v_A_kmpm = v_A_kmh / 60.0   # km/h → km/min 변환
    return {
        "area_size":               100,
        "n_points_per_zone":       int(N_P_zone),
        "point_value_range":       [1, 10],
        "targets_per_point_range": [T_PP, T_PP],
        "target_value_range":      [1, 10],
        "target_window_range":     [int(dk_min), int(dk_min + DK_WIDTH)],
        "target_offset_range":     [2, 8],
        "n_recon_uav":    N_R_FIXED,
        "recon_start":    [0, 0],
        "recon_speed":    V_R,
        "recon_max_dist": F_R,
        "attack_speed":    v_A_kmpm,
        "attack_max_dist": F_A,
        "attack_weapons":  W_FIXED,
        "random_seed":     int(seed),
    }


# ══════════════════════════════════════════════════════════════════
# 2. 반응변수 추출
# ══════════════════════════════════════════════════════════════════
def extract_metrics(res: dict) -> dict:
    hits     = res["hits"]
    explored = res["explored"]
    K        = res["K"]
    P        = res["P"]
    A        = res["A"]

    n_hit = len(hits)
    n_exp = len(explored)
    sv    = sum(res["k_val"][k] for k in hits)
    hr    = n_hit / len(K) if K else 0.0
    rr    = n_exp / len(P) if P else 0.0

    K_a = res.get("K_a", {})
    bind_count = sum(
        1 for a in A
        if sum(1 for k in K_a.get(a, []) if k in hits) >= W_FIXED
    )
    weapon_bind = bind_count / len(A) if A else 0.0

    return {
        "sv":          sv,
        "hr":          round(hr, 4),
        "rr":          round(rr, 4),
        "n_hit":       n_hit,
        "n_exp":       n_exp,
        "weapon_bind": round(weapon_bind, 4),
    }


# ══════════════════════════════════════════════════════════════════
# 3. 단일 실험점 실행 (병렬 안전)
# ══════════════════════════════════════════════════════════════════
def run_one(v_A_kmh: float, dk_min: float, N_P_zone: int,
            seed: int, tmp_dir: str) -> dict:
    cfg = make_cfg(v_A_kmh, dk_min, N_P_zone, seed)
    xls = os.path.join(tmp_dir,
          f"_doe_vA{int(v_A_kmh)}_dk{int(dk_min)}_NP{N_P_zone}_s{seed}.xlsx")

    df_p, df_t, df_r, df_a = generate_dataframes(cfg)
    save_excel(df_p, df_t, df_r, df_a, xls)

    t0 = time.time()
    try:
        res = build_and_solve(xls, time_limit=TIME_LIMIT, msg=False)
    except Exception as e:
        return {"_error": str(e)}
    finally:
        if os.path.exists(xls):
            try: os.remove(xls)
            except: pass

    elapsed = round(time.time() - t0, 2)
    m = extract_metrics(res)
    m.update({
        "sec":        elapsed,
        "solver_sec": round(res.get("elapsed", 0), 2),
        "status":     res["status"],
    })
    return m


def _run_one_wrapper(job: dict) -> dict:
    """ProcessPoolExecutor용 래퍼"""
    m = run_one(job["v_A_kmh"], job["dk_min"], job["N_P_zone"],
                job["seed"],    job["tmp_dir"])
    if m is None:
        m = {"_error": "None returned"}
    m["_job"] = job
    return m


# ══════════════════════════════════════════════════════════════════
# 4. 설계 행렬 (2³ + 중심점 3회)
# ══════════════════════════════════════════════════════════════════
def build_design() -> pd.DataFrame:
    rows = []
    # 2³ 완전 요인점 (8점)
    for combo in itertools.product([-1, 1], repeat=3):
        rows.append({
            "type":     "factorial",
            "v_A_kmh":  LEVELS["v_A_kmh"][0]  if combo[0]==-1 else LEVELS["v_A_kmh"][2],
            "dk_min":   LEVELS["dk_min"][0]    if combo[1]==-1 else LEVELS["dk_min"][2],
            "N_P_zone": LEVELS["N_P_zone"][0]  if combo[2]==-1 else LEVELS["N_P_zone"][2],
            "cV": combo[0], "cD": combo[1], "cP": combo[2],
        })
    # 중심점 3회
    for _ in range(3):
        rows.append({
            "type":     "center",
            "v_A_kmh":  LEVELS["v_A_kmh"][1],
            "dk_min":   LEVELS["dk_min"][1],
            "N_P_zone": LEVELS["N_P_zone"][1],
            "cV": 0, "cD": 0, "cP": 0,
        })
    df = pd.DataFrame(rows).reset_index(drop=True)
    df.insert(0, "run_id", range(1, len(df)+1))
    return df


# ══════════════════════════════════════════════════════════════════
# 5. 전체 실험 루프 (병렬 실행)
# ══════════════════════════════════════════════════════════════════
def _progress_monitor(done_counter: list, total: int,
                      running_jobs: list, stop_event: threading.Event):
    frames = ["|", "/", "-", "\\"]
    i = 0
    while not stop_event.is_set():
        done = done_counter[0]
        pct  = done / total * 100
        sys.stdout.write(
            f"\r  {frames[i%4]}  진행: {done:2d}/{total}  ({pct:5.1f}%)  "
            f"동시실행: {len(running_jobs)}개  [{time.strftime('%H:%M:%S')}]   "
        )
        sys.stdout.flush()
        i += 1
        time.sleep(0.3)
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


def _restart_progress(done_counter, total, running_jobs):
    stop_evt = threading.Event()
    t = threading.Thread(
        target=_progress_monitor,
        args=(done_counter, total, running_jobs, stop_evt),
        daemon=True)
    t.start()
    return stop_evt, t


def run_doe(design: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    tmp_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    total   = len(design) * len(R_SEEDS)
    records = []

    print(f"\n{'═'*70}")
    print(f"  Kill Chain DoE v4 — 4구역 모델  [병렬 실행]")
    print(f"  {len(design)}점 × {len(R_SEEDS)}시드 = {total}회 실험")
    print(f"  고정: v_R={V_R:.3f}km/min  F_R={F_R:.0f}km  "
          f"F_A={F_A:.0f}km  W={W_FIXED}  T_pp={T_PP}  n_R={N_R_FIXED}")
    print(f"  인자: v_A∈{LEVELS['v_A_kmh']}km/h  "
          f"dk_min∈{LEVELS['dk_min']}min  "
          f"NP_zone∈{LEVELS['N_P_zone']}  (Δk~U[dk,dk+{DK_WIDTH}])")
    print(f"  병렬: N_WORKERS={N_WORKERS}  스레드/솔버={N_THREADS_PER_SOLVER}  "
          f"(총 {N_WORKERS*N_THREADS_PER_SOLVER}/{_cpu}코어 사용)")
    print(f"  솔버 시간 제한: {TIME_LIMIT}s/회")
    print(f"{'═'*70}")

    jobs = []
    for _, row in design.iterrows():
        for seed in R_SEEDS:
            jobs.append({
                "run_id":   int(row.run_id),
                "type":     row.type,
                "v_A_kmh":  float(row.v_A_kmh),
                "dk_min":   float(row.dk_min),
                "N_P_zone": int(row.N_P_zone),
                "cV": row.cV, "cD": row.cD, "cP": row.cP,
                "seed":     seed,
                "tmp_dir":  tmp_dir,
            })

    done_counter  = [0]
    running_jobs  = []
    stop_progress, prog_thread = _restart_progress(
        done_counter, total, running_jobs)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_job = {
            executor.submit(_run_one_wrapper, job): job
            for job in jobs
        }
        running_jobs.extend(list(future_to_job.values()))

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            running_jobs.remove(job)
            done_counter[0] += 1

            stop_progress.set(); prog_thread.join()

            try:
                m = future.result()
            except Exception as e:
                print(f"\n  ERROR [vA={job['v_A_kmh']},dk={job['dk_min']},NP={job['N_P_zone']},s={job['seed']}]: {e}")
                stop_progress, prog_thread = _restart_progress(
                    done_counter, total, running_jobs)
                continue

            if "_error" in m:
                print(f"\n  SKIP [vA={job['v_A_kmh']},dk={job['dk_min']},NP={job['N_P_zone']},s={job['seed']}]: {m['_error']}")
                stop_progress, prog_thread = _restart_progress(
                    done_counter, total, running_jobs)
                continue

            status_short = "OPT" if m["status"] == "Optimal" else "BF"
            print(f"\n  [{done_counter[0]:3d}/{total}] "
                  f"vA={int(job['v_A_kmh']):4d}km/h  "
                  f"Δk={int(job['dk_min']):2d}min  "
                  f"NP={job['N_P_zone']:2d}  s={job['seed']}  "
                  f"→ sv={m['sv']:5.0f} hr={m['hr']:.3f} "
                  f"rr={m['rr']:.3f} wb={m['weapon_bind']:.2f} "
                  f"[{status_short}] {m['sec']}s")

            records.append({
                "run_id":   job["run_id"], "type":     job["type"],
                "v_A_kmh":  job["v_A_kmh"], "dk_min": job["dk_min"],
                "N_P_zone": job["N_P_zone"],
                "cV": job["cV"], "cD": job["cD"], "cP": job["cP"],
                "seed":     job["seed"],
                **{k: v for k, v in m.items() if not k.startswith("_")},
            })

            stop_progress, prog_thread = _restart_progress(
                done_counter, total, running_jobs)

    stop_progress.set(); prog_thread.join()

    try: os.rmdir(tmp_dir)
    except OSError: pass

    df_raw = pd.DataFrame(records)
    df_raw = df_raw.sort_values(["run_id", "seed"]).reset_index(drop=True)
    csv_path = os.path.join(out_dir, "doe_raw.csv")
    df_raw.to_csv(csv_path, index=False)
    print(f"\n  원시 데이터 ({len(df_raw)}행) → {csv_path}")
    return df_raw


# ══════════════════════════════════════════════════════════════════
# 6. 통계 분석
# ══════════════════════════════════════════════════════════════════
def analyze(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_ff  = df_raw[df_raw["type"] == "factorial"].copy()
    fcodes = [fi[1] for fi in FACTOR_INFO]
    fnames = [fi[2] for fi in FACTOR_INFO]

    # ── 주효과 & 교호작용 효과 크기 ──────────────────────────────
    effects = {}
    for resp, rname in RESP_INFO:
        row = {}
        # 주효과
        for fc, fn in zip(fcodes, fnames):
            g = df_ff.groupby(fc)[resp].mean()
            row[fn] = round(g.get(1, 0) - g.get(-1, 0), 4)
        # 2인자 교호작용
        for i in range(len(fcodes)):
            for j in range(i+1, len(fcodes)):
                fi2, fj2 = fcodes[i], fcodes[j]
                col2 = f"_int_{fi2}{fj2}"
                df_ff[col2] = df_ff[fi2] * df_ff[fj2]
                g = df_ff.groupby(col2)[resp].mean()
                row[f"{fnames[i]}×{fnames[j]}"] = round(g.get(1,0)-g.get(-1,0), 4)
        # 3인자 교호작용
        col3 = "_int_cVcDcP"
        df_ff[col3] = df_ff["cV"] * df_ff["cD"] * df_ff["cP"]
        g = df_ff.groupby(col3)[resp].mean()
        row[f"{fnames[0]}×{fnames[1]}×{fnames[2]}"] = round(g.get(1,0)-g.get(-1,0), 4)
        effects[rname] = row
    df_effects = pd.DataFrame(effects).T

    # ── ANOVA ─────────────────────────────────────────────────────
    anova_rows = []
    for resp, rname in [("sv", "Strike Value"), ("hr", "Hit Ratio")]:
        for fc, fn in zip(fcodes, fnames):
            grps = [g[resp].values for _, g in df_ff.groupby(fc)]
            F, p = stats.f_oneway(*grps)
            sig  = "***" if p<0.001 else ("**" if p<0.01 else
                   ("*"   if p<0.05  else "n.s."))
            lo = df_ff[df_ff[fc]==-1][resp].mean()
            hi = df_ff[df_ff[fc]== 1][resp].mean()
            anova_rows.append({
                "Response": rname, "Factor": fn,
                "Low_mean":  round(lo,  3),
                "High_mean": round(hi,  3),
                "Effect":    round(hi-lo, 3),
                "F-stat":    round(F,   3),
                "p-value":   round(p,   4),
                "sig": sig,
            })
    df_anova = pd.DataFrame(anova_rows)

    # ── 곡률 검정 ─────────────────────────────────────────────────
    df_c  = df_raw[df_raw["type"]=="center"]["sv"]
    _, pc = stats.ttest_ind(df_c, df_ff["sv"])

    print("\n[주효과 & 교호작용 효과 크기]\n")
    print(df_effects.to_string())
    print("\n\n[ANOVA — Strike Value & Hit Ratio]\n")
    for _, r in df_anova.iterrows():
        print(f"  [{r.Response:13s}] {r.Factor:35s}  "
              f"Low={r.Low_mean:7.3f}  High={r.High_mean:7.3f}  "
              f"Δ={r.Effect:+7.3f}  F={r['F-stat']:7.3f}  "
              f"p={r['p-value']:.4f}  {r.sig}")
    print(f"\n  곡률 검정 (sv)  p={pc:.4f}  "
          f"{'** 비선형 존재' if pc<0.05 else 'n.s. — 선형 근사 적합'}")
    return df_effects, df_anova


# ══════════════════════════════════════════════════════════════════
# 7. 시각화 (6종)
# ══════════════════════════════════════════════════════════════════
def _save(fig, path, label):
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {label}: {os.path.basename(path)}")


def visualize(df_raw: pd.DataFrame, df_effects: pd.DataFrame, out_dir: str):
    df_ff = df_raw[df_raw["type"]=="factorial"].copy()
    plt.rcParams.update({"axes.titlesize":10, "axes.labelsize":9,
                          "xtick.labelsize":8, "ytick.labelsize":8})

    v_A_levs = [LEVELS["v_A_kmh"][0],  LEVELS["v_A_kmh"][2]]
    dk_levs  = [LEVELS["dk_min"][0],   LEVELS["dk_min"][2]]
    np_levs  = [LEVELS["N_P_zone"][0], LEVELS["N_P_zone"][2]]

    # ── Fig1: 주효과도 (SV + HR, 3인자) ──────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ri, (resp, rtitle, col) in enumerate([("sv","Strike Value",PA[0]),
                                               ("hr","Hit Ratio",   PA[1])]):
        for fi, (fname, fcode, ftitle) in enumerate(FACTOR_INFO):
            ax   = axes[ri][fi]
            lo_d = df_ff[df_ff[fcode]==-1][resp]
            hi_d = df_ff[df_ff[fcode]== 1][resp]
            ys   = [lo_d.mean(), hi_d.mean()]
            sems = [lo_d.sem(),  hi_d.sem()]
            levs = [LEVELS[fname][0], LEVELS[fname][2]]
            ax.errorbar([-1,1], ys, yerr=sems, fmt="o-", color=col,
                        lw=2.5, ms=9, capsize=6,
                        markerfacecolor="white", markeredgewidth=2.5)
            ax.fill_between([-1,1],
                            [ys[0]-sems[0], ys[1]-sems[1]],
                            [ys[0]+sems[0], ys[1]+sems[1]],
                            alpha=0.12, color=col)
            ax.set_xticks([-1,1]); ax.set_xticklabels([str(l) for l in levs])
            ax.set_title(f"{rtitle} vs {ftitle.split('(')[0].strip()}", pad=5)
            if fi==0: ax.set_ylabel(rtitle)
            ax.grid(alpha=0.3)
            ax.annotate(f"Δ={ys[1]-ys[0]:+.2f}",
                        xy=(0, max(ys)+max(sems)*0.5), ha="center",
                        fontsize=9, color=col, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    fig.suptitle("Main Effect Plots — Kill Chain DoE v4 (4-Zone)\n"
                 f"v_R={V_R:.2f}km/min  W={W_FIXED}(fixed)  "
                 f"n_R={N_R_FIXED}(fixed)  |  "
                 "2³ Full Factorial + 3 center, 6 replicates",
                 fontsize=10, fontweight="bold", y=1.01)
    _save(fig, os.path.join(out_dir,"fig1_main_effects.png"), "Fig1 주효과")

    # ── Fig2: 교호작용도 (3쌍) ───────────────────────────────────
    def iplot(ax, fx, fy, xname, yname, xlevs, ylevs, resp="hr"):
        for yv, lc, lbl in [(-1, PA[1], f"{yname}=Low({ylevs[0]})"),
                              ( 1, PA[0], f"{yname}=High({ylevs[1]})")]:
            sub = df_ff[df_ff[fy]==yv].groupby(fx)[resp]
            xs  = sorted(sub.groups.keys())
            ys  = [sub.get_group(x).mean() for x in xs]
            se  = [sub.get_group(x).sem()  for x in xs]
            ax.errorbar(xs, ys, yerr=se, fmt="o-", color=lc, lw=2.2, ms=8,
                        capsize=5, markerfacecolor="white",
                        markeredgewidth=2, label=lbl)
        ax.set_xticks([-1,1])
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    iplot(axes[0], "cV","cD", "v_A","Δk_min", v_A_levs, dk_levs)
    axes[0].set_xticklabels([f"{v}\nkm/h" for v in v_A_levs])
    axes[0].set_title("v_A × Δk_min\n(Hit Ratio)", fontweight="bold")
    axes[0].set_ylabel("Hit Ratio")

    iplot(axes[1], "cV","cP", "v_A","N_P_zone", v_A_levs, np_levs)
    axes[1].set_xticklabels([f"{v}\nkm/h" for v in v_A_levs])
    axes[1].set_title("v_A × N_P_zone\n(Hit Ratio)", fontweight="bold")
    axes[1].set_ylabel("Hit Ratio")

    iplot(axes[2], "cD","cP", "Δk_min","N_P_zone", dk_levs, np_levs)
    axes[2].set_xticklabels([f"{v}\nmin" for v in dk_levs])
    axes[2].set_title("Δk_min × N_P_zone\n(Hit Ratio)", fontweight="bold")
    axes[2].set_ylabel("Hit Ratio")

    fig.suptitle("Interaction Effect Plots (Hit Ratio)",
                 fontsize=12, fontweight="bold")
    _save(fig, os.path.join(out_dir,"fig2_interactions.png"), "Fig2 교호작용")

    # ── Fig3: 파레토 차트 ─────────────────────────────────────────
    def shorten(k):
        return (k.replace("Attack Speed (v_A, km/h)", "v_A")
                 .replace("Time Window min (Δ_k, min)", "Δk")
                 .replace("Points/Zone (N_P)", "N_P"))

    fig, axes_p = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (rname, col) in zip(axes_p, [("Strike Value",PA[0]),
                                          ("Hit Ratio",   PA[1])]):
        row  = df_effects.loc[rname].abs().sort_values(ascending=True)
        lbls = [shorten(k) for k in row.index]
        bcol = [col if "×" not in k else "#90A4AE" for k in row.index]
        ax.barh(range(len(row)), row.values, color=bcol,
                edgecolor="white", lw=0.5)
        ax.set_yticks(range(len(row))); ax.set_yticklabels(lbls, fontsize=8)
        ax.set_xlabel("|Effect Size|")
        ax.set_title(f"Pareto — {rname}", fontweight="bold")
        thr = 2 * row.mean()
        ax.axvline(thr, color="red", lw=1.8, linestyle="--",
                   label=f"2σ ({thr:.3f})", alpha=0.8)
        for i, (k, v) in enumerate(row.items()):
            if v >= thr:
                ax.text(v+0.001, i, "★", fontsize=10, color=col, va="center")
        ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Pareto Charts (★=|Effect|>2σ)", fontsize=12, fontweight="bold")
    _save(fig, os.path.join(out_dir,"fig3_pareto.png"), "Fig3 파레토")

    # ── Fig4: 반응표면 히트맵 (v_A × Δk, NP별 HR) ────────────────
    fig, axes_h = plt.subplots(1, 2, figsize=(13, 5))
    for ai, (np_val, np_code, ax) in enumerate(
            zip([LEVELS["N_P_zone"][0], LEVELS["N_P_zone"][2]],
                [-1, 1], axes_h)):
        sub = df_ff[df_ff["cP"]==np_code]
        xv  = sorted(sub["cV"].unique())
        yv  = sorted(sub["cD"].unique())
        xlbls = [f"{int(LEVELS['v_A_kmh'][0] if x==-1 else LEVELS['v_A_kmh'][2])}" for x in xv]
        ylbls = [f"{int(LEVELS['dk_min'][0]  if y==-1 else LEVELS['dk_min'][2])}"  for y in yv]
        M = np.full((len(yv), len(xv)), np.nan)
        for i, y in enumerate(yv):
            for j, x in enumerate(xv):
                s = sub[(sub["cV"]==x) & (sub["cD"]==y)]["hr"]
                if len(s): M[i,j] = s.mean()
        im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(xv))); ax.set_xticklabels(xlbls, fontsize=9)
        ax.set_yticks(range(len(yv))); ax.set_yticklabels(ylbls, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Hit Ratio")
        for i in range(len(yv)):
            for j in range(len(xv)):
                ax.text(j, i, f"{M[i,j]:.3f}",
                        ha="center", va="center",
                        fontsize=12, fontweight="bold", color="black")
        ax.set_xlabel("v_A (km/h)"); ax.set_ylabel("Δk_min (min)")
        ax.set_title(f"Hit Ratio — NP_zone={np_val} (W/NP={W_FIXED/np_val:.2f})",
                     fontweight="bold")
    fig.suptitle("Hit Ratio Heat Map (v_A × Δk_min) by N_P_zone",
                 fontsize=12, fontweight="bold")
    _save(fig, os.path.join(out_dir,"fig4_heatmaps.png"), "Fig4 반응표면")

    # ── Fig5: 박스플롯 (SV, 3인자) ───────────────────────────────
    fig, axes_b = plt.subplots(1, 3, figsize=(13, 5))
    for fi, (fname, fcode, ftitle) in enumerate(FACTOR_INFO):
        ax  = axes_b[fi]
        lo  = df_ff[df_ff[fcode]==-1]["sv"].values
        hi  = df_ff[df_ff[fcode]== 1]["sv"].values
        bp  = ax.boxplot([lo, hi], patch_artist=True, widths=0.5,
                         medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor("#BBDEFB")
        bp["boxes"][1].set_facecolor("#EF9A9A")
        levs = [LEVELS[fname][0], LEVELS[fname][2]]
        units = {"v_A_kmh":"km/h", "dk_min":"min", "N_P_zone":"개"}
        unit = units.get(fname, "")
        ax.set_xticks([1,2])
        ax.set_xticklabels([f"Low\n({levs[0]}{unit})",
                             f"High\n({levs[1]}{unit})"], fontsize=8)
        ax.set_title(ftitle, fontweight="bold")
        if fi==0: ax.set_ylabel("Strike Value")
        ax.grid(axis="y", alpha=0.3)
        t, p = stats.ttest_ind(lo, hi)
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "n.s."))
        ymax = max(max(lo), max(hi))
        ax.text(1.5, ymax * 1.03, f"t={t:.2f}\n{sig}",
                ha="center", fontsize=8, color="navy", fontweight="bold")
    fig.suptitle("Strike Value Distribution by Factor Level",
                 fontsize=12, fontweight="bold")
    _save(fig, os.path.join(out_dir,"fig5_boxplots.png"), "Fig5 박스플롯")

    # ── Fig6: 곡률 검정 ──────────────────────────────────────────
    df_c  = df_raw[df_raw["type"]=="center"]["sv"]
    df_fp = df_raw[df_raw["type"]=="factorial"]["sv"]
    _, pc = stats.ttest_ind(df_c, df_fp)
    fig, ax = plt.subplots(figsize=(6, 5))
    means = [df_fp.mean(), df_c.mean()]
    sems  = [df_fp.sem(),  df_c.sem()]
    ax.bar(["Factorial mean", "Center mean"],
           means, color=[PA[0], PA[2]], edgecolor="white", width=0.5)
    ax.errorbar([0, 1], means, yerr=sems,
                fmt="none", color="black", capsize=8, lw=2)
    ax.set_ylabel("Strike Value")
    ax.set_title("Curvature Test — Strike Value", fontweight="bold")
    txt = (f"p={pc:.4f}  "
           f"{'n.s.  (linear OK)' if pc>=0.05 else '**  (curvature detected)'}")
    ax.text(0.5, max(means)*1.04, txt,
            ha="center", fontsize=10, color="navy", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir,"fig6_curvature.png"), "Fig6 곡률검정")


# ══════════════════════════════════════════════════════════════════
# 8. 결과 Excel 저장
# ══════════════════════════════════════════════════════════════════
def save_doe_excel(df_raw, df_effects, df_anova, out_dir):
    df_ff  = df_raw[df_raw["type"]=="factorial"]
    df_agg = (df_ff.groupby(["v_A_kmh","dk_min","N_P_zone"])
              [["sv","hr","rr","n_hit","weapon_bind","sec","solver_sec"]]
              .agg(["mean","std"]).round(4))
    df_agg.columns = ["_".join(c) for c in df_agg.columns]
    df_agg = df_agg.reset_index()

    path = os.path.join(out_dir, "doe_results.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_raw.to_excel(writer,     sheet_name="raw_data",   index=False)
        df_agg.to_excel(writer,     sheet_name="aggregated", index=False)
        df_effects.to_excel(writer, sheet_name="effects",    index=True)
        df_anova.to_excel(writer,   sheet_name="anova",      index=False)
    print(f"  DoE 결과 Excel: {path}")


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    OUT = os.path.join(BASE_DIR, "doe_results")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    design = build_design()

    print(f"\n{'═'*70}")
    print(f"  Kill Chain DoE v4 — 실험 설계 행렬")
    print(f"{'═'*70}")
    print(design[["run_id","type","v_A_kmh","dk_min","N_P_zone"]].to_string(index=False))
    print(f"\n  ▸ 고정: W={W_FIXED}발  T_pp={T_PP}  n_R={N_R_FIXED}")
    print(f"  ▸ 고정: v_R={V_R:.3f}km/min ({V_R*60:.0f}km/h)  F_R={F_R:.0f}km  F_A={F_A:.0f}km")
    print(f"  ▸ 인자 A: v_A ∈ {LEVELS['v_A_kmh']} km/h")
    print(f"  ▸ 인자 B: Δk_min ∈ {LEVELS['dk_min']} min  (Δk~U[dk,dk+{DK_WIDTH}])")
    print(f"  ▸ 인자 C: N_P_zone ∈ {LEVELS['N_P_zone']} 개/구역  (W/NP: {W_FIXED}/{LEVELS['N_P_zone'][0]}~{W_FIXED}/{LEVELS['N_P_zone'][2]})")
    print(f"  ▸ 핵심 가설: v_A↓+Δk↓ → HR 급락 / NP↑ → W/NP 비율↓ → 무장 binding 심화")
    print(f"  ▸ 반복 시드: {R_SEEDS} ({len(R_SEEDS)}개, 검정력 80% 확보)")
    print(f"  ▸ 병렬: N_WORKERS={N_WORKERS}  CPU={_cpu}코어")
    print(f"  ▸ 솔버 시간 제한: {TIME_LIMIT}s/회")
    print(f"  ▸ 총 {len(design)}점 × {len(R_SEEDS)}시드 = "
          f"{len(design)*len(R_SEEDS)}회  "
          f"예상: ~{len(design)*len(R_SEEDS)*60//(60*N_WORKERS)}~"
          f"{len(design)*len(R_SEEDS)*TIME_LIMIT//(60*N_WORKERS)}분")

    print("\n[DoE 실험 실행]")
    df_raw = run_doe(design, OUT)

    print("\n[통계 분석]")
    df_effects, df_anova = analyze(df_raw)

    print("\n[시각화]")
    visualize(df_raw, df_effects, OUT)

    save_doe_excel(df_raw, df_effects, df_anova, OUT)

    print(f"\n{'═'*70}")
    print(f"  완료 | 총 소요시간: {time.time()-t0:.1f}초 | 출력: {OUT}")
    print(f"{'═'*70}")

