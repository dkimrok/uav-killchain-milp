#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_missing_cells.py
=====================
단일코어 샌드박스에서 못 푼 Experiment II 하드코너 셀(v_A=300, Δk=3, N_P≥10)을
멀티코어 노트북에서 최적해로 채우는 단독 실행 스크립트.

필요 파일(같은 폴더):
    - killchain_solver_fix2.py   (Big-M 타이트닝 + 정찰 고정 솔버)
    - killchain_auto.py          (인스턴스 생성기)
    - killchain_solver.py        (선택; 없어도 동작)

설치:
    pip install pulp pandas numpy openpyxl
    # matplotlib은 불필요 (그리기 전용)

실행:
    python fill_missing_cells.py
    # (옵션) 셀당 제한시간(초) 지정:  python fill_missing_cells.py 600

출력:
    missing_cells_results.csv   (기존 expII_6seed.csv 와 같은 열 구조)
        -> 기존 CSV 뒤에 이어붙이면 완전 균형 6시드가 됩니다.

주의:
    - 솔버는 자동으로 모든 코어를 사용합니다(threads=os.cpu_count()).
    - 결과의 재현성: 인스턴스 생성 cfg가 샌드박스 실행과 동일하므로,
      이미 풀린 시드를 다시 돌리면 같은 HR이 나와야 합니다(검증용).
"""
import sys, os, csv, time
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

fx  = _load("killchain_solver_fix2", os.path.join(HERE, "killchain_solver_fix2.py"))
auto = _load("killchain_auto",       os.path.join(HERE, "killchain_auto.py"))
generate_dataframes = auto.generate_dataframes
save_excel          = auto.save_excel

# ── 실험 상수 (샌드박스 실행과 동일) ──────────────────────────────
W = 8  # weapon load

def cfg(v_A, dk, NP, seed):
    return {
        "area_size": 100,
        "n_points_per_zone": int(NP),
        "point_value_range": [1, 10],
        "targets_per_point_range": [1, 1],
        "target_value_range": [1, 10],
        "target_window_range": [int(dk), int(dk + 2)],
        "target_offset_range": [2, 8],
        "n_recon_uav": 4,
        "recon_start": [0, 0],
        "recon_speed": 260 / 60,
        "recon_max_dist": 260 * 4.6,
        "attack_speed": v_A / 60.0,
        "attack_max_dist": 1110 * 1.5,
        "attack_weapons": W,
        "random_seed": int(seed),
    }

# ── 채워야 할 셀(누락/미증명) ─────────────────────────────────────
#   (v_A, Δk_min, N_P) : [채울 시드들]
#   - N_P=10, v300/dk3 : 7, 137, 314  (s42/s99/s21 은 이미 최적)
#   - N_P=12, v300/dk3 : 99, 7, 21, 137, 314  (99는 feasible→최적 재확인)
MISSING = {
    (300, 3, 10): [7, 137, 314],
    (300, 3, 12): [99, 7, 21, 137, 314],
}

# expII_6seed.csv 와 동일한 열
HEADER = ["v_A", "dk_min", "N_P", "WoverNP", "seed", "status", "obj_value",
          "sv", "hr", "rr", "n_hit", "n_K", "n_explored", "n_P", "weapon_bind", "sec"]


def solve_cell(v_A, dk, NP, seed, time_limit):
    xls = os.path.join(HERE, f"_inst_{v_A}_{dk}_{NP}_{seed}.xlsx")
    dfp, dft, dfr, dfa = generate_dataframes(cfg(v_A, dk, NP, seed))
    save_excel(dfp, dft, dfr, dfa, xls)
    t0 = time.time()
    r = fx.build_and_solve(xls, time_limit=time_limit, msg=False)
    sec = round(time.time() - t0, 1)
    try:
        os.remove(xls)
    except OSError:
        pass
    hits = r["hits"]; A = r["A"]; K_a = r.get("K_a", {})
    sv = sum(r["k_val"][k] for k in hits)
    hr = len(hits) / len(r["K"]) if r["K"] else 0.0
    rr = len(r["explored"]) / len(r["P"]) if r["P"] else 0.0
    bind = (sum(1 for a in A if sum(1 for k in K_a.get(a, []) if k in hits) >= W) / len(A)) if A else 0.0
    row = [v_A, dk, NP, round(W / NP, 3), seed, r["status"], round(r["obj_value"], 1),
           sv, round(hr, 4), round(rr, 4), len(hits), len(r["K"]),
           len(r["explored"]), len(r["P"]), round(bind, 4), sec]
    return r["status"], hr, row


def main():
    time_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    out = os.path.join(HERE, "missing_cells_results.csv")
    print(f"cores = {os.cpu_count()} | time_limit/cell = {time_limit}s")
    print(f"output -> {out}\n")
    write_header = not os.path.exists(out)
    f = open(out, "a", newline="")
    wtr = csv.writer(f)
    if write_header:
        wtr.writerow(HEADER)
    done, opt = 0, 0
    for (v_A, dk, NP), seeds in MISSING.items():
        for seed in seeds:
            status, hr, row = solve_cell(v_A, dk, NP, seed, time_limit)
            done += 1
            tag = "OPTIMAL" if status == "Optimal" else status.upper()
            if status == "Optimal":
                opt += 1
            print(f"[{done:2d}] vA{v_A} dk{dk} NP{NP} s{seed:<4d} -> {tag:9s} HR={hr:.4f} ({row[-1]}s)")
            if status in ("Optimal", "Feasible") and row[6] > 0:
                wtr.writerow(row); f.flush()
    f.close()
    print(f"\n완료: {done}개 중 {opt}개 최적해. 결과: {out}")
    print("기존 expII_6seed.csv 뒤에 이어붙이면 완전 균형 6시드가 됩니다.")


if __name__ == "__main__":
    main()
