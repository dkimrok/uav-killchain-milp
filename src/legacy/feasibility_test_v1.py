#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feasibility_test.py  —  D2 option B (cumulative time, limited waiting) 타당성 테스트
===================================================================================
목적 : N_P=12 결합 최적화(정찰 라우팅 + 공격 스케줄링)가 실용적 시간 안에
        풀리는지 판정한다.  안 풀리면 Exp II 상위수준을 12 -> 10 으로 낮춘다.

배치 : joint_model.py, killchain_auto.py, killchain_solver_fix2.py 와 같은 폴더
설치 : pip install pulp highspy pandas numpy openpyxl
실행 : python feasibility_test.py                # 기본: 셀당 1800초
       python feasibility_test.py 3600           # 셀당 제한시간(초) 지정
       python feasibility_test.py 1800 8         # 제한시간, 스레드 수

판정 기준 (셀당 제한시간 T 기준)
  PASS  : N_P=12 네 코너 모두 Optimal,  최악 셀 <= 0.5*T      -> 설계 유지
  MARGIN: N_P=12 Optimal 이지만 최악 셀 > 0.5*T               -> 시드 축소 검토
  FAIL  : N_P=12 중 하나라도 미해결                            -> N_P 상위수준 10 으로
"""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_model as J

TL      = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 4)
SEED    = 42
CORNERS = [(300, 3), (300, 9), (1110, 3), (1110, 9)]
OUT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feasibility_test.csv")

def main():
    print(f"threads={THREADS}  time_limit={TL}s/cell  seed={SEED}")
    print(f"s_service=1.0 min  W_max=2.0 min\n")
    rows = []
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N_P", "v_A", "dk", "seed", "status", "sv", "hr", "sec"])
        for NP in (6, 10, 12):                       # 6 = 사전점검, 10 = 대안, 12 = 본안
            for (v, dk) in CORNERS:
                t0 = time.time()
                try:
                    r = J.solve(v, dk, NP, SEED, time_limit=TL,
                                threads=THREADS, wmax=2.0, s_svc=1.0)
                    st, sv, hr = r["status"], r["sv"], r["hr"]
                except Exception as e:
                    st, sv, hr = f"ERROR:{e}", 0, 0
                sec = round(time.time() - t0, 1)
                w.writerow([NP, v, dk, SEED, st, sv, round(hr, 4), sec]); f.flush()
                rows.append((NP, v, dk, st, sec))
                print(f"  N_P={NP:2d}  v_A={v:4d}  dk={dk}  ->  {st:12s} "
                      f"sv={sv:6.0f} hr={hr:.4f}  [{sec}s]")
            # 조기 종료: 이 수준이 전부 실패면 상위 수준은 볼 필요 없음
            lvl = [r for r in rows if r[0] == NP]
            if all(r[3] != "Optimal" for r in lvl):
                print(f"  -> N_P={NP} 전부 미해결. 상위 수준 생략.")
                break
    print(f"\n결과: {OUT}\n" + "="*64)
    for NP in (12, 10, 6):
        lvl = [r for r in rows if r[0] == NP]
        if not lvl: continue
        ok  = all(r[3] == "Optimal" for r in lvl)
        wst = max(r[4] for r in lvl)
        verdict = "PASS" if (ok and wst <= 0.5*TL) else ("MARGIN" if ok else "FAIL")
        print(f"  N_P={NP:2d}: {verdict:6s}  (최악 셀 {wst}s / 제한 {TL}s)")
    print("="*64)
    print("PASS   -> 설계 그대로 (Exp II: N_P 6/10/12)")
    print("MARGIN -> 설계 유지하되 시드 6->4 축소 또는 야간 배치 실행")
    print("FAIL   -> Exp II 상위수준을 12 -> 10 으로 (W/N_P=0.8, 무장 바인딩 관찰 가능)")

if __name__ == "__main__":
    main()
