# Changelog

## 1.0.0 — 2026-08-09

Version of record accompanying the submitted manuscript. The model differs
substantively from the earlier development versions retained in `src/legacy/`;
the changes below are recorded because they alter reported results.

### Model

- **Cumulative attack arrival times.** `T_{a,j} >= T_{a,i} + s + d_ij / v_A`
  replaces a bound written against the target's own window opening. Under the
  earlier form the reconnaissance completion time cancelled from the
  feasibility condition, arrival times were not monotone in sequence position,
  and the joint problem decomposed into two separable problems. Servicing time
  `s = 1 min` is new.
- **Bounded loitering.** `T_{a,j} <= T_{a,i} + s + d_ij / v_A + W_max` with
  `W_max = 2 min`. A sensitivity run over `W_max` in {0, 1, 2, 5, 30} minutes
  leaves both responses unchanged at every design point tested
  (`results/sensitivity_wmax.csv`), so the cap is not binding in this regime.
- **Zone-assigned reconnaissance.** Each zone is surveyed by one dedicated
  reconnaissance UAV. Visit sequence and completion times remain endogenous.
- **Departure-linking inequality.** `sum_k h_{a,k} <= |K_a| * sum_k y_{a,0_a,k}`.
  Implied by the flow and MTZ constraints but very weak in the linear
  relaxation.
- **Zone-corner holding points** at (0,0), (100,0), (0,100), (100,100), replacing
  a single common depot at the origin.

### Objective and solver

- **Two-phase lexicographic objective** replaces the single Big-M objective.
  Phase 1 maximises weighted reconnaissance coverage; Phase 2 maximises strike
  value subject to attaining the Phase-1 optimum.
- **Cardinality tie-break** `eps * sum h`, `eps = 1/(|K|+1)`. Without it the hit
  ratio is not uniquely determined, because the objective maximises value while
  the reported response is a count.
- **`gapRel = 1e-6`, `gapAbs = 0`.**
- **Termination status read from HiGHS directly.** PuLP maps the HiGHS
  `kTimeLimit` code to `LpStatusOptimal`, so a run that merely exhausted its
  time budget is indistinguishable from a certified optimum at the modelling
  layer. Every row of `results/exp*_raw.csv` carries a `proven` flag.

### Experiment design

- Experiment II high level moved from `N_P = 12` to `N_P = 10`, placing the
  centre point at `W / N_P = 1.000` exactly.
- Centre point executed once per seed rather than three times, since repeated
  runs at identical settings are byte-identical duplicates.
- 108 runs total (54 per experiment), all certified optimal.

### Analysis

- `make_tables.py` is the single source of every number in the paper's results
  chapter. Main effects use the saturated 2^3 factorial model tested against
  pure error; curvature uses Montgomery's test with pooled pure error, applied
  identically to both responses in both experiments.
- Missing cells are never imputed. Cell-mean substitution shrinks pure error and
  inflates F.
