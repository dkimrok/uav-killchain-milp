# Provenance archive — earlier formulations, March–June 2026

These files are the raw output of **superseded model versions** and are retained
so that the development history of the study is auditable. They are **not** the
results reported in the paper; those are in `results/expI_raw.csv` and
`results/expII_raw.csv`, produced by the current formulation in `src/`.

All four files were produced by the non-cumulative attack-time model now kept in
`src/legacy/killchain_solver.py`, under which arrival times did not accumulate
along the strike sequence and the reconnaissance completion time cancelled from
the engagement feasibility condition.

| File | Design | Runs | Status |
|---|---|---|---|
| `expI_original_2026-03-20.csv` | `v_A` × `Δk_min` × `N_P ∈ {4,6,8}` | 66 | all Optimal |
| `expII_original_np14_2026-03-20.csv` | `v_A` × `Δk_min` × `N_P ∈ {6,10,14}` | 66 | 61 Optimal, 5 Feasible |
| `nR_screening_2026-03-12.csv` | `n_R` × `N_P` × `Δk_min` | 33 | 12 Optimal, 21 Feasible |
| `expII_6seed_2026-06-29.csv` | `v_A` × `Δk_min` × `N_P ∈ {6,10,12}` | 75 | 70 Optimal, 4 Not Solved, 1 Feasible |

## What each file shows

**`expI_original_2026-03-20.csv`.** The origin of the hit-ratio figures reported
in the first draft of the paper. Under the earlier model the low-speed
narrow-window cell means were 0.2083 at `N_P = 4` and 0.5104 at `N_P = 8`,
averaging 0.3594. Re-solving the same instances with a cardinality tie-break on
the objective returns 0.2083 and 0.5052. Strike value is identical in both
cases; only the count differs. This is the alternative-optimum problem that
motivated the `ε Σ h` term in the current objective: the model maximised value
while the reported response was a count, so the hit ratio was not uniquely
determined by the formulation.

**`expII_original_np14_2026-03-20.csv`.** Contains a target density,
`N_P = 14`, that lies outside the range of the main design. All eighteen
non-degenerate corner runs return a hit ratio of 0.571429 with zero variance,
exactly `W / N_P = 8 / 14`. The current formulation reproduces this value; see
`results/extension_np14.csv`.

**`nR_screening_2026-03-12.csv`.** The screening design behind the claim, made in
an early draft, that reconnaissance UAV count is not a significant factor. The
data does not support that claim and is retained as a record of why it was
withdrawn:

- 21 of the 33 runs terminated at a 60-second limit without certifying
  optimality, so their objective values are incumbents rather than optima.
- The centre point was executed three times per seed. For seed 42 the three
  runs returned strike values of 10, 39 and 39. The instance generator and the
  model are both deterministic, so identical settings must give identical
  results; the spread is an artefact of time-limited termination.
- At `N_P = 6` both `n_R = 2` and `n_R = 4` return a hit ratio of exactly 1.000.
  The response is at its ceiling, so no factor can register an effect there.
- At `N_P = 14` the mean hit ratio is 0.1042 at `n_R = 2` and 0.0000 at
  `n_R = 4`. More sensors producing a worse result is not an operational finding;
  the larger model simply yielded a poorer incumbent within the same time limit.

The reconnaissance-side question is addressed in the current paper by comparing
optimised reconnaissance sequencing against a fixed nearest-neighbour tour at
equal asset count, which isolates scheduling from quantity.

**`expII_6seed_2026-06-29.csv`.** An intermediate Experiment II run under the
recon-fixing decomposition (`src/legacy/killchain_solver_fix2.py`). Four cells
did not solve and one returned a feasible-only incumbent. Superseded.

## Column note

The two March files use the earlier driver schema
(`run_id, type, <factors>, cN/cV, cP, cD, seed, sv, hr, rr, n_hit, n_exp,
weapon_bind, sec, solver_sec, status`). The June file uses the schema of
`results/legacy/expII_final.csv`. Neither carries a `proven` flag; in the
current pipeline that flag is written by reading the solver's termination status
directly, because the modelling layer maps a time-limit termination to an
optimal status.
