# Superseded code

These files are retained so that the development history of the model is
auditable. **They do not produce the results reported in the paper.** Use the
modules in `../` instead.

| File | Why it was superseded |
|---|---|
| `killchain_solver.py` | Attack arrival times were bounded against each target's own window opening rather than against the predecessor's arrival. Travel time did not accumulate along the strike sequence, so `D_p` cancelled from the feasibility condition and the reconnaissance schedule had no effect on strike feasibility. |
| `killchain_solver_fix2.py` | Fixed reconnaissance to a nearest-neighbour full-coverage tour, justified by that same cancellation. Once the arrival recursion is written cumulatively the cancellation no longer holds, so the decomposition is not exact. Also folded reconnaissance coverage into a single Big-M objective, which inflated the objective to 10^4–10^5 and made the solver's default relative gap of 1e-4 worth several units of strike value. |
| `killchain_doe.py` | Ran the centre point three times per seed. Because the generator and the solver are both deterministic those runs are byte-identical duplicates, not independent replications, and counting them inflates the pure-error degrees of freedom. Its `analyze()` used one-way ANOVA per factor, which uses a different error term from the saturated factorial model reported in the paper. |
| `fill_missing_cells.py` | Ad hoc re-run helper for cells that the superseded solver could not certify. No longer needed: every cell certifies in seconds under the current formulation. |
| `joint_model_v1_free_recon.py` | Reconnaissance UAVs free to cross zone boundaries. Correct but intractable: the low-speed narrow-window corner did not certify within 1,800 s on 16 threads, and other corners returned no feasible solution at all. |
| `joint_model_v2_no_entry_cut.py` | Zone-assigned reconnaissance but without the departure-linking inequality. In zones containing no entry-feasible target the optimum is zero, and proving it required more than 665,000 nodes; the current model certifies the same cell in under 20 s. |
| `feasibility_test_v1.py`, `feasibility_test_v2.py` | Tractability probes for the two superseded joint models. |
