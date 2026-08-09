#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_extension.py — extension validation at a single target-density level
========================================================================
Runs the four factorial corners (v_A in {300, 1110} x dk_min in {3, 9}) at one
N_P level, across the standard seed set. Used to confirm that the structural
ceiling min(1, W/N_P) continues to hold outside the factor ranges of the main
design, without altering the main design itself.

Output schema matches results/exp*_raw.csv, so make_tables.py and the figure
scripts can read it directly.

Checkpointed: rerunning skips (cell x seed) rows already recorded as proven.

    python run_extension.py --np 14 --out ../results --threads 16
    python run_extension.py --np 14 --out ../results --time-limit 3600
"""
import sys, os, csv, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_model as J
from killchain_auto import generate_dataframes

SEEDS = [42, 99, 314, 7, 21, 137]
CORNERS = [(300, 3), (300, 9), (1110, 3), (1110, 9)]
HEADER = ["exp", "v_A", "dk_min", "N_P", "WoverNP", "seed", "ptype", "status",
          "proven", "max_gap", "obj", "sv", "hr", "rr", "n_hit", "n_K",
          "n_exp", "n_P", "weapon_bind", "sec"]


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["v_A"], r["dk_min"], r["N_P"], r["seed"])
                for r in csv.DictReader(f) if r.get("proven") == "True"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, required=True, help="reconnaissance points per zone")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--time-limit", type=int, default=1800, help="seconds per zone subproblem")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"extension_np{a.np}.csv")
    seeds = [int(s) for s in a.seeds.split(",")]
    W = 8
    have = done_keys(path)
    # easy corners first, low-speed narrow-window last
    cells = sorted(CORNERS, key=lambda c: (c[0] == 300 and c[1] == 3))
    todo = [(v, d, s) for (v, d) in cells for s in seeds
            if (str(v), str(d), str(a.np), str(s)) not in have]
    print(f"Extension validation | N_P={a.np}  W/N_P={W/a.np:.4f}  "
          f"corners={len(cells)} seeds={len(seeds)} | done={len(have)} to run={len(todo)}")
    print(f"threads={a.threads}  limit={a.time_limit}s per zone subproblem  -> {path}\n")

    new = not os.path.exists(path)
    t0 = time.time()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        for i, (v, d, s) in enumerate(todo, 1):
            t = time.time()
            try:
                r = J.solve(v, d, a.np, s, time_limit=a.time_limit,
                            threads=a.threads, recon_mode="zone")
            except Exception as e:
                r = dict(status=f"ERROR:{e}", proven=False, max_gap=float("nan"),
                         sv=0, hr=0, n_hit=0, n_K=0)
            dp, _, _, _ = generate_dataframes(J.cfg(v, d, a.np, s))
            n_P = len(dp)
            w.writerow(["EXT", v, d, a.np, round(W / a.np, 4), s, "factorial", r["status"],
                        r["proven"], r["max_gap"], "", r["sv"], round(r["hr"], 6), 1.0,
                        r["n_hit"], r["n_K"], n_P, n_P,
                        round(min(1.0, r["n_hit"] / (4 * W)), 4), round(time.time() - t, 1)])
            f.flush()
            flag = "" if r["proven"] else "   <-- NOT PROVEN"
            print(f"[{i:3d}/{len(todo)}] v_A={v:4d} dk={d} seed={s:3d}  {r['status']:14s} "
                  f"sv={r['sv']:6.0f} hr={r['hr']:.4f}  [{time.time()-t:7.1f}s]{flag}")

    print(f"\n완료: {len(todo)}회, {(time.time()-t0)/60:.1f}분")
    import pandas as pd
    df = pd.read_csv(path)
    bad = df[df.proven != True]
    print(f"기록 {len(df)}행 중 미증명 {len(bad)}행")
    if len(bad) == 0:
        g = df.groupby(["v_A", "dk_min"]).hr.agg(["mean", "std", "size"])
        print(f"\nceiling W/N_P = {W/a.np:.6f}")
        print(g.round(6).to_string())
    else:
        print("미증명 행은 --time-limit 을 늘려 재실행하십시오 (완료분은 건너뜁니다).")


if __name__ == "__main__":
    main()
