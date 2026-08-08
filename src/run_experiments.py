#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiments.py  —  Exp I / Exp II 본실험 드라이버
======================================================
설계
  Exp I  : v_A {300,600,1110} x dk_min {3,6,9} x N_P {4,6,8}
  Exp II : v_A {300,600,1110} x dk_min {3,6,9} x N_P {6,8,10}
  각 실험 = 2^3 요인점 8셀 + 중심점 1셀,  시드 6개 -> 54회

중심점은 시드당 1회만 실행한다. 인스턴스 생성기와 솔버가 모두 결정론적이므로
동일 시드 재실행은 완전 중복이며 순수오차 자유도를 부풀린다.

정직한 상태 기록
  PuLP 는 HiGHS 의 kTimeLimit 을 LpStatusOptimal 로 매핑한다. 본 드라이버는
  HighsModelStatus 를 직접 읽어 status / proven / max_gap 을 기록한다.
  proven=False 인 행은 분석에서 제외해야 한다 (make_tables.py 가 경고함).

체크포인트 / 재개
  매 실행마다 CSV 에 append 하고, 재실행 시 이미 있는 (셀 x 시드) 는 건너뛴다.
  중단되어도 그대로 다시 실행하면 이어서 진행한다.

실행
  python run_experiments.py --exp I  --out ../results
  python run_experiments.py --exp II --out ../results --threads 16
  python run_experiments.py --exp II --out ../results --cells 300,3,10   # 특정 셀만
"""
import sys, os, csv, time, argparse, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_model as J
from killchain_auto import generate_dataframes

SEEDS   = [42, 99, 314, 7, 21, 137]
LEVELS  = {"I":  dict(v_A=(300, 600, 1110), dk=(3, 6, 9), N_P=(4, 6, 8)),
           "II": dict(v_A=(300, 600, 1110), dk=(3, 6, 9), N_P=(6, 8, 10))}
HEADER  = ["exp", "v_A", "dk_min", "N_P", "WoverNP", "seed", "ptype", "status",
           "proven", "max_gap", "obj", "sv", "hr", "rr", "n_hit", "n_K",
           "n_exp", "n_P", "weapon_bind", "sec"]


def design(exp):
    L = LEVELS[exp]
    lo_v, ct_v, hi_v = L["v_A"]; lo_d, ct_d, hi_d = L["dk"]; lo_p, ct_p, hi_p = L["N_P"]
    pts = [(v, d, p, "factorial")
           for v in (lo_v, hi_v) for d in (lo_d, hi_d) for p in (lo_p, hi_p)]
    pts.append((ct_v, ct_d, ct_p, "center"))
    return pts


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["v_A"], r["dk_min"], r["N_P"], r["seed"])
                for r in csv.DictReader(f) if r.get("proven") == "True"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["I", "II"], required=True)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--time-limit", type=int, default=1800, help="구역(subproblem)당 상한 초")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--cells", default="", help="'v_A,dk,N_P' 세미콜론 구분. 지정 시 그 셀만")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    path  = os.path.join(a.out, f"exp{a.exp}_raw.csv")
    seeds = [int(s) for s in a.seeds.split(",")]
    pts   = design(a.exp)
    if a.cells:
        want = {tuple(int(x) for x in c.split(",")) for c in a.cells.split(";")}
        pts  = [p for p in pts if (p[0], p[1], p[2]) in want]

    # 쉬운 셀부터: 저속·협소창 코너를 마지막에 배치해 조기 피드백을 얻는다
    pts.sort(key=lambda p: (p[0] == min(LEVELS[a.exp]["v_A"]) and p[1] == min(LEVELS[a.exp]["dk"]),
                            p[2]))
    have = done_keys(path)
    new  = not os.path.exists(path)
    todo = [(v, d, p, t, s) for (v, d, p, t) in pts for s in seeds
            if (str(v), str(d), str(p), str(s)) not in have]
    print(f"Experiment {a.exp} | design points={len(pts)} seeds={len(seeds)} "
          f"| already done={len(have)} | to run={len(todo)}")
    print(f"threads={a.threads}  limit={a.time_limit}s per zone subproblem  -> {path}\n")

    t_all = time.time()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        for i, (v, d, p, ptype, s) in enumerate(todo, 1):
            t0 = time.time()
            try:
                r = J.solve(v, d, p, s, time_limit=a.time_limit,
                            threads=a.threads, recon_mode="zone")
            except Exception as e:
                r = dict(status=f"ERROR:{e}", proven=False, max_gap=float("nan"),
                         sv=0, hr=0, n_hit=0, n_K=0, sec=0)
            dp, dt, _, _ = generate_dataframes(J.cfg(v, d, p, s))
            n_P = len(dp)
            W   = 8
            w.writerow([a.exp, v, d, p, round(W / p, 4), s, ptype, r["status"],
                        r["proven"], r["max_gap"], "", r["sv"], round(r["hr"], 6),
                        1.0, r["n_hit"], r["n_K"], n_P, n_P,
                        round(min(1.0, r["n_hit"] / (4 * W)), 4), round(time.time() - t0, 1)])
            f.flush()
            flag = "" if r["proven"] else "   <-- NOT PROVEN"
            print(f"[{i:3d}/{len(todo)}] v_A={v:4d} dk={d} N_P={p:2d} seed={s:3d} "
                  f"{ptype:9s} {r['status']:22s} sv={r['sv']:6.0f} hr={r['hr']:.4f} "
                  f"[{time.time()-t0:7.1f}s]{flag}")
    el = time.time() - t_all
    print(f"\n완료: {len(todo)}회, {el/60:.1f}분")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows if r["proven"] != "True"]
    print(f"기록 {len(rows)}행 중 미증명 {len(bad)}행")
    for r in bad:
        print(f"  v_A={r['v_A']} dk={r['dk_min']} N_P={r['N_P']} seed={r['seed']} -> {r['status']}")
    if bad:
        print("\n미증명 행은 --time-limit 을 늘려 재실행하십시오 (완료분은 자동으로 건너뜁니다).")
    else:
        print(f"\n전 셀 증명 완료.  다음: python make_tables.py {path} --exp {a.exp} --out ../tables")


if __name__ == "__main__":
    main()
