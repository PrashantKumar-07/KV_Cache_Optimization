# Final figures, renumbered (18 November 2026)

This folder renders the final figures in `figures_final/` under their current names (Fig. 1–9) and checks them. Everything runs on CPU from the frozen, verified raw records; it takes about 15 seconds.

```bash
cd results/figures_20261118_renumbered
python render.py            # Fig. 1-4, 6-9 + supplement, from 10,318 verified predictions
python render_timing.py     # Fig. 5, from the verified timing run
python verify_numeric.py    # independent raw-record -> plotted-value check (no renderer import)
python verify_timing.py     # independent check of the 30 latency medians
python renumbering_proof.py # byte identity with the October outputs and figures_final/
```

| New name | Old (October) name | Source |
|---|---|---|
| fig1_accuracy_equivram | fig1_accuracy_equivram | full evaluation, hot 1,024 |
| fig2_accuracy_fair | fig7_accuracy_fair | full evaluation, hot 992 |
| fig3_write_savings | fig3_write_savings | full evaluation, hot 992 |
| fig4_footprint | fig11_footprint | full evaluation endpoint snapshots |
| fig5_decode_scaling | fig4_decode_scaling | `results/audit_20261016_timing` |
| fig6_modeled_traffic_time | fig2_latency_comparison | counters × assumed prices |
| fig7_energy | fig12_energy | counters × assumed prices |
| fig8_migration_profile | fig6_migration_profile | full evaluation, hot 1,024 |
| fig9_heatmap | fig14_heatmap | full evaluation, hot 1,024 |

## What changed relative to October

- `render.py`, `verify_numeric.py`, `render_timing.py` and `verify_timing.py` are copies of the October scripts in `results/audit_20261016_current_figures/` and `results/audit_20261016_timing/figures/`. Only the output names, sidecar file names and paths changed, plus the following:
  - Style helpers come from `experiments/figure_style.py`, a verbatim subset of the legacy `plot_figures_v2.py`; only its comments' figure numbers were updated.
  - The renderer no longer asserts the hashes of the removed legacy plotting scripts. `renumbering_proof.py` instead checks that those scripts are preserved byte-exact in `results/full_evaluation_20261015/source.tar.gz` and match `plan.json`.
  - The unverified Fig. 8/15 are no longer listed as "pending"; they were removed.
- **Result:** all 20 PDF/PNG outputs, the CSV tables and the raw-record hash list are byte-identical to the October outputs (`renumbering_proof.json`). No value changed.
