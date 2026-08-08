#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_tables.py — Chapter IV 표·통계 단일 생성기
================================================
논문 Chapter IV 의 모든 수치를 **하나의 문서화된 방법**으로 재생성한다.
목적: "논문에 실린 수치 = 이 스크립트의 출력" 을 보장하여 재현성 문제를 제거한다.

입력  : 원자료 CSV. 필수 열
          v_A, dk_min, N_P, seed, status, sv, hr
        (n_hit, n_K, rr, sec 등 추가 열은 무시)
실행  : python make_tables.py results/expI_raw.csv  --exp I  --out tables/
        python make_tables.py results/expII_raw.csv --exp II --out tables/
출력  : tables/expX_main_effects.csv   (Table IV-4 / IV-7)
        tables/expX_cell_means.csv     (Table IV-5 / IV-8)
        tables/expX_curvature.csv
        tables/expX_report.txt         (사람이 읽는 요약 + 방법 명시)

────────────────────────────────────────────────────────────────────────
통계 방법 (논문 본문에 이 문구를 그대로 실을 것)
────────────────────────────────────────────────────────────────────────
1) 주효과·교호작용
   2^3 완전요인 배치(반복 n)의 **포화 요인 모형**으로 분석한다.
   효과크기 = (상위수준 평균) − (하위수준 평균)
   SS_effect = (contrast)^2 / (n · 2^k),  자유도 1
   포화모형이므로 잔차제곱합 = 순수오차제곱합(SS_PE), 자유도 2^k(n−1)
   F = MS_effect / MS_PE
   ※ 이전 코드의 stats.f_oneway (1원 ANOVA) 는 다른 오차항을 쓰므로 사용하지 않는다.

2) 곡률검정 (Montgomery, Design and Analysis of Experiments, 9th ed.)
   SS_curv = n_F·n_C·(ȳ_F − ȳ_C)^2 / (n_F + n_C),  자유도 1
   오차항 = 요인점 순수오차 + 중심점 순수오차 (풀링)
   F = SS_curv / MS_PE_pooled
   ※ 두 표본 t-검정(등분산/Welch 혼용) 은 사용하지 않는다.
   ※ SV 와 HR 에 **동일한** 방법을 적용한다.

3) 중심점
   중심점은 시드당 1회만 집계한다. 동일 시드로 3회 반복 실행한 결과는
   결정론적 솔버에서 완전 중복이므로 독립 반복이 아니다.

4) 결측 처리
   결측 셀은 **대체하지 않는다**. 결측이 있으면 경고하고 불균형 상태로 보고한다.
   (셀평균 대체는 순수오차를 축소시켜 F 를 부풀린다.)
"""
import sys, os, argparse
import numpy as np
import pandas as pd
from scipy import stats

RESPONSES = [("sv", "Strike Value"), ("hr", "Hit Ratio")]
FACTORS   = [("cV", "v_A"), ("cD", "dk"), ("cP", "N_P")]


def code_design(df, levels):
    """요인점/중심점 분류 및 ±1 코딩. levels = {'v_A':(lo,ctr,hi), ...}"""
    lo_v, ct_v, hi_v = levels["v_A"]
    lo_d, ct_d, hi_d = levels["dk"]
    lo_p, ct_p, hi_p = levels["N_P"]
    d = df.copy()
    is_f = (d.v_A.isin([lo_v, hi_v]) & d.dk_min.isin([lo_d, hi_d]) & d.N_P.isin([lo_p, hi_p]))
    is_c = (d.v_A.eq(ct_v) & d.dk_min.eq(ct_d) & d.N_P.eq(ct_p))
    d["ptype"] = np.where(is_f, "factorial", np.where(is_c, "center", "other"))
    d["cV"] = np.where(d.v_A == hi_v, 1, np.where(d.v_A == lo_v, -1, 0))
    d["cD"] = np.where(d.dk_min == hi_d, 1, np.where(d.dk_min == lo_d, -1, 0))
    d["cP"] = np.where(d.N_P == hi_p, 1, np.where(d.N_P == lo_p, -1, 0))
    return d


def dedup_center(d):
    """중심점은 (seed) 당 1행만 남긴다."""
    c = d[d.ptype == "center"]
    dup = len(c) - c.seed.nunique()
    if dup > 0:
        d = pd.concat([d[d.ptype != "center"], c.drop_duplicates(subset=["seed"])],
                      ignore_index=True)
    return d, dup


def pure_error(ff, resp, keys=("cV", "cD", "cP")):
    ss, dfree = 0.0, 0
    for _, g in ff.groupby(list(keys)):
        y = g[resp].values
        ss += float(((y - y.mean()) ** 2).sum()); dfree += len(y) - 1
    return ss, dfree


def factorial_anova(ff, resp):
    """포화 2^3 요인모형. 효과크기 + F + p."""
    n_cells = ff.groupby(["cV", "cD", "cP"]).size()
    if n_cells.nunique() != 1:
        balanced, n = False, None
    else:
        balanced, n = True, int(n_cells.iloc[0])
    N = len(ff)
    ss_pe, df_pe = pure_error(ff, resp)
    ms_pe = ss_pe / df_pe if df_pe > 0 else np.nan
    terms = {"cV": ["cV"], "cD": ["cD"], "cP": ["cP"],
             "cV:cD": ["cV", "cD"], "cV:cP": ["cV", "cP"], "cD:cP": ["cD", "cP"],
             "cV:cD:cP": ["cV", "cD", "cP"]}
    rows = []
    for name, cols in terms.items():
        sign = np.ones(N, dtype=int)
        for c in cols:
            sign = sign * ff[c].values
        y = ff[resp].values
        hi, lo = y[sign == 1], y[sign == -1]
        effect = hi.mean() - lo.mean()
        contrast = y[sign == 1].sum() - y[sign == -1].sum()
        ss = contrast ** 2 / N
        F = ss / ms_pe if ms_pe and ms_pe > 0 else np.nan
        p = float(stats.f.sf(F, 1, df_pe)) if np.isfinite(F) else np.nan
        rows.append(dict(Term=name, Low_mean=round(lo.mean(), 4),
                         High_mean=round(hi.mean(), 4), Effect=round(effect, 4),
                         SS=round(ss, 4), F=round(F, 3) if np.isfinite(F) else np.nan,
                         p=round(p, 5) if np.isfinite(p) else np.nan,
                         sig=("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s.")
                         if np.isfinite(p) else "n/a"))
    return pd.DataFrame(rows), dict(ss_pe=ss_pe, df_pe=df_pe, ms_pe=ms_pe,
                                    balanced=balanced, n=n)


def curvature(ff, cen, resp):
    """Montgomery 곡률검정. 풀링된 순수오차 사용."""
    if len(cen) == 0:
        return None
    yF, yC = ff[resp].values, cen[resp].values
    nF, nC = len(yF), len(yC)
    ss_f, df_f = pure_error(ff, resp)
    ss_c, df_c = float(((yC - yC.mean()) ** 2).sum()), nC - 1
    df_pe = df_f + df_c
    ms_pe = (ss_f + ss_c) / df_pe if df_pe > 0 else np.nan
    ss_curv = nF * nC * (yF.mean() - yC.mean()) ** 2 / (nF + nC)
    F = ss_curv / ms_pe if ms_pe and ms_pe > 0 else np.nan
    p = float(stats.f.sf(F, 1, df_pe)) if np.isfinite(F) else np.nan
    return dict(response=resp, factorial_mean=round(yF.mean(), 4),
                center_mean=round(yC.mean(), 4), n_F=nF, n_C=nC,
                SS_curvature=round(ss_curv, 4), MS_pure_error=round(ms_pe, 5),
                df_error=df_pe, F=round(F, 3) if np.isfinite(F) else np.nan,
                p=round(p, 5) if np.isfinite(p) else np.nan,
                verdict=("significant curvature" if p < .05 else "n.s. - linear model adequate")
                if np.isfinite(p) else "n/a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--exp", choices=["I", "II"], required=True)
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = pd.read_csv(a.csv)
    need = {"v_A", "dk_min", "N_P", "seed", "sv", "hr"}
    miss = need - set(raw.columns)
    if miss:
        sys.exit(f"필수 열 누락: {sorted(miss)}")
    # 요인 수준은 데이터에서 자동 감지한다 (하드코딩 시 설계 변경마다 어긋남)
    levels = {}
    for key, col in [("v_A", "v_A"), ("dk", "dk_min"), ("N_P", "N_P")]:
        u = sorted(raw[col].unique())
        if len(u) != 3:
            sys.exit(f"{col}: 수준이 3개가 아닙니다 -> {u}")
        levels[key] = (u[0], u[1], u[2])
    print("감지된 요인 수준: " + ", ".join(f"{k}={v}" for k, v in levels.items()))
    d = code_design(raw, levels)
    d, dup = dedup_center(d)
    ff  = d[d.ptype == "factorial"].copy()
    cen = d[d.ptype == "center"].copy()
    oth = d[d.ptype == "other"]

    L = []
    L.append(f"Experiment {a.exp}  |  source: {os.path.basename(a.csv)}")
    L.append(f"factorial rows={len(ff)}  center rows={len(cen)}  off-design rows={len(oth)}")
    if dup:
        L.append(f"  [note] 중심점 중복 {dup}행 제거 (동일 시드 재실행 = 완전 중복)")
    if len(oth):
        L.append(f"  [warn] 설계 밖 조건 {len(oth)}행은 분석에서 제외됨:")
        for _, g in oth.groupby(["v_A", "dk_min", "N_P"]):
            L.append(f"           v_A={g.v_A.iloc[0]} dk={g.dk_min.iloc[0]} N_P={g.N_P.iloc[0]} ({len(g)}행)")
    if "status" in d.columns:
        bad = d[d.status.astype(str) != "Optimal"]
        if len(bad):
            L.append(f"  [warn] 비최적 상태 {len(bad)}행 포함 -> gapRel=0 으로 재실행 권장")

    cells = (ff.groupby(["N_P", "v_A", "dk_min"])
               .agg(n=("hr", "size"), HR=("hr", "mean"), HR_sd=("hr", "std"),
                    SV=("sv", "mean"), SV_sd=("sv", "std")).round(4).reset_index())
    if len(cen):
        cc = (cen.groupby(["N_P", "v_A", "dk_min"])
                 .agg(n=("hr", "size"), HR=("hr", "mean"), HR_sd=("hr", "std"),
                      SV=("sv", "mean"), SV_sd=("sv", "std")).round(4).reset_index())
        cells = pd.concat([cells, cc], ignore_index=True)
    cells.to_csv(os.path.join(a.out, f"exp{a.exp}_cell_means.csv"), index=False)
    L.append("\n[Cell means]\n" + cells.to_string(index=False))

    allf = []
    for resp, label in RESPONSES:
        tab, info = factorial_anova(ff, resp)
        tab.insert(0, "Response", label)
        allf.append(tab)
        if not info["balanced"]:
            L.append(f"\n  [warn] {label}: 셀별 반복수 불균형 -> 결측 대체 없이 그대로 분석함")
        L.append(f"\n[{label}] saturated 2^3 ANOVA  "
                 f"(MS_pure_error={info['ms_pe']:.5f}, df={info['df_pe']}, n/cell={info['n']})")
        L.append(tab.drop(columns=["Response"]).to_string(index=False))
    pd.concat(allf, ignore_index=True).to_csv(
        os.path.join(a.out, f"exp{a.exp}_main_effects.csv"), index=False)

    cur = [c for c in (curvature(ff, cen, r) for r, _ in RESPONSES) if c]
    if cur:
        cdf = pd.DataFrame(cur)
        cdf.to_csv(os.path.join(a.out, f"exp{a.exp}_curvature.csv"), index=False)
        L.append("\n[Curvature test — Montgomery, pooled pure error]")
        L.append(cdf.to_string(index=False))
    else:
        L.append("\n[Curvature test] 중심점 없음 - 생략")

    rep = "\n".join(L)
    with open(os.path.join(a.out, f"exp{a.exp}_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep + "\n")
    print(rep)
    print(f"\n-> {a.out}/exp{a.exp}_*.csv, exp{a.exp}_report.txt")


if __name__ == "__main__":
    main()
