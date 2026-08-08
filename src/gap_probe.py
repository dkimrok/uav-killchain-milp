#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gap_probe.py  —  "시간을 더 주면 풀리는가?" 를 판정하는 진단 도구
==================================================================
한 셀을 제한시간을 늘려가며 반복 실행하고, 매번
  incumbent(현재 최선해) / dual bound(상계) / gap  을 기록한다.

판정
  gap 이 단계마다 뚜렷이 줄어든다        -> 시간을 늘리면 풀린다
  gap 이 정체하거나 dual bound 가 안 내려온다 -> 시간을 늘려도 소용없다 (모형을 바꿔야 함)
  incumbent 조차 못 찾는다               -> 완화가 너무 약함. 시간 문제 아님

실행
  python gap_probe.py                          # 기본 (300,3,N_P=10) zone 모드
  python gap_probe.py 300 3 10 zone 16         # v_A dk N_P mode threads
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_model as J
import pulp, highspy

V    = int(sys.argv[1]) if len(sys.argv) > 1 else 300
DK   = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NP   = int(sys.argv[3]) if len(sys.argv) > 3 else 10
MODE = sys.argv[4] if len(sys.argv) > 4 else "zone"
THR  = int(sys.argv[5]) if len(sys.argv) > 5 else (os.cpu_count() or 4)
LIMITS = [60, 300, 900, 1800, 3600]
SEED = 42

_orig = J._status
def probe_status(prob, tl, el):
    lab, gap = _orig(prob, tl, el)
    try:
        info = prob.solverModel.getInfo()
        probe_status.zones.append((lab, abs(float(info.objective_function_value)),
                                   abs(float(info.mip_dual_bound)), float(info.mip_gap),
                                   int(info.mip_node_count)))
    except Exception:
        pass
    return lab, gap
probe_status.zones = []
J._status = probe_status

print(f"cell: v_A={V}  dk={DK}  N_P={NP}  mode={MODE}  threads={THR}  seed={SEED}")
print(f"{'limit':>7} {'status':>24} {'incumbent':>11} {'bound':>11} {'gap':>10} {'nodes':>10} {'sec':>8}")
print("-" * 88)
prev = None
for tl in LIMITS:
    t0 = time.time()
    probe_status.zones = []
    r = J.solve(V, DK, NP, SEED, time_limit=tl, threads=THR, recon_mode=MODE)
    el = round(time.time() - t0, 1)
    zs = probe_status.zones or [("n/a", float("nan"), float("nan"), float("nan"), 0)]
    worst = max(zs, key=lambda t: (t[3] if t[3] == t[3] else -1))   # 최악 gap 구역
    lab, inc, bnd, gap, nodes = worst
    print(f"{tl:>7} {lab:>24} {inc:>11.2f} {bnd:>11.2f} "
          f"{gap:>10.4f} {nodes:>10} {el:>8}   (구역별: "
          + ', '.join(f'{t[0].split("(")[0]}' for t in zs) + ")")
    if r["proven"]:
        print("\n-> 증명 완료. 이 제한시간이면 충분합니다.")
        break
    if prev is not None and prev < 1e30 and gap < 1e30:
        red = (prev - gap) / prev if prev > 0 else 0
        if red < 0.05:
            print(f"\n-> gap 감소율 {red*100:.1f}% (< 5%). "
                  "시간을 늘려도 수렴하지 않습니다. 모형을 바꾸십시오.")
            break
    prev = gap
