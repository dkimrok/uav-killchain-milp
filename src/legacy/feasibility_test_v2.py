#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feasibility_test_v2.py  —  타당성 테스트 (개정판)
==================================================
v1 대비 변경
  1. joint_model_v2 사용: 구역 전담 정찰(recon_mode='zone')로 문제를 4개 구역
     부분문제로 분해.  D_p 는 여전히 내생(경로/타이밍 최적화)이다.
  2. 정직한 상태 보고.  PuLP 는 HiGHS 의 kTimeLimit 을 LpStatusOptimal 로
     매핑하므로 'Optimal' 라벨을 신뢰할 수 없다.  HighsModelStatus 와 MIP gap
     을 직접 읽어 Optimal / TimeLimit 을 구분한다.
  3. --mode free 로 v1(구역 넘나드는 정찰) 과 비교 가능.

실행
  python feasibility_test_v2.py                    # 셀당 1800초, 전 코어
  python feasibility_test_v2.py 3600 16            # 제한시간, 스레드
  python feasibility_test_v2.py 1800 16 free       # 정찰 자유이동 비교용

판정
  PASS   : 해당 N_P 네 코너 전부 status=Optimal (증명됨) 이고 최악 <= 0.5*T
  MARGIN : 전부 Optimal 이나 최악 > 0.5*T
  FAIL   : 하나라도 TimeLimit / 미증명
"""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_model_v3 as J

TL      = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 4)
MODE    = sys.argv[3] if len(sys.argv) > 3 else "zone"
SEED    = 42
CORNERS = [(300, 3), (300, 9), (1110, 3), (1110, 9)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"feasibility_v2_{MODE}.csv")

def main():
    print(f"recon_mode={MODE}  threads={THREADS}  time_limit={TL}s/cell  seed={SEED}")
    print(f"s_service=1.0 min  W_max=2.0 min  gapRel=1e-6\n")
    rows = []
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N_P","v_A","dk","seed","mode","status","proven","max_gap","sv","hr","sec"])
        for NP in (6, 10, 12):
            for (v, dk) in CORNERS:
                try:
                    r = J.solve(v, dk, NP, SEED, time_limit=TL, threads=THREADS,
                                recon_mode=MODE, wmax=2.0, s_svc=1.0)
                except Exception as e:
                    r = dict(status=f"ERROR:{e}", proven=False, max_gap=float("nan"),
                             sv=0, hr=0, sec=0)
                w.writerow([NP, v, dk, SEED, MODE, r["status"], r["proven"],
                            r["max_gap"], r["sv"], round(r["hr"], 4), r["sec"]]); f.flush()
                rows.append((NP, r["proven"], r["sec"]))
                print(f"  N_P={NP:2d} v_A={v:4d} dk={dk} -> {r['status']:34s} "
                      f"sv={r['sv']:6.0f} hr={r['hr']:.4f} [{r['sec']}s]")
            if not any(p for _, p, _ in [x for x in rows if x[0] == NP]):
                print(f"  -> N_P={NP} 전부 미증명. 상위 수준 생략.")
                break
    print(f"\n결과: {OUT}\n" + "=" * 64)
    for NP in (12, 10, 6):
        lv = [r for r in rows if r[0] == NP]
        if not lv: continue
        ok = all(p for _, p, _ in lv); wst = max(s for _, _, s in lv)
        print(f"  N_P={NP:2d}: {'PASS' if (ok and wst<=0.5*TL) else ('MARGIN' if ok else 'FAIL'):6s}"
              f"  (최악 {wst}s / 제한 {TL}s)")
    print("=" * 64)

if __name__ == "__main__":
    main()
