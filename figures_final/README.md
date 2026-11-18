> **Paper-comparison restriction:** The main adapted-baseline figures document real functional-control results; they are not a SOTA comparison. The official GQA SnapKV (39.52) and upstream SnapKV (39.80) both score above TieredKV (38.50 / 38.81). Do not cite the adapted-method win as SOTA superiority.

# Final figures (renumbered 18 November 2026)

All nine figures below come from verified runs. Numbers are contiguous and follow the order in which the presentation uses them.

| # | File | What it shows | Evidence |
|---|------|---------------|----------|
| 1 | `fig1_accuracy_equivram` | LongBench accuracy at the same hot budget (5 methods, 6 tasks) | full evaluation, 10,318 predictions |
| 2 | `fig2_accuracy_fair` | Accuracy at the same attended rows (992 hot + 32 cold = 1,024) | full evaluation |
| 3 | `fig3_write_savings` | Paid vs avoided logical cold-tier writes during decode | full evaluation |
| 4 | `fig4_footprint` | Configured capacity and observed KV payload storage | full evaluation (endpoint snapshots) |
| 5 | `fig5_decode_scaling` | Measured median decode latency vs context length | fresh timing benchmark (4,500 forwards) |
| 6 | `fig6_modeled_traffic_time` | Summed **modeled** KV traffic time (not measured latency) | full evaluation counters × assumed prices |
| 7 | `fig7_energy` | **Modeled** KV traffic energy at assumed per-bit prices | full evaluation counters × assumed prices |
| 8 | `fig8_migration_profile` | TieredKV policy actions (promote / demote / reactivate / drop) per task | full evaluation |
| 9 | `fig9_heatmap` | Per-task score heatmap: TieredKV (hot 1,024) vs full cache and 3 adapted baselines | full evaluation |

Supplemental: `supplemental/supplement_faithful_accuracy` compares against the faithful (expanded-cache) SnapKV and shows both wins and losses.

Architecture diagram: `../figures/tieredkv_architecture_v2.svg` (PDF/PNG beside it). It is drawn from the frozen evaluation code: hot tier 992 rows, cold tier 2,048 rows, ≤32 exposed cold rows per step, write-once victim rule.

## Old → new numbers

**Current renderer:** `../results/figures_20261118_renumbered/` writes these exact file names. Run `render.py` and `render_timing.py`, then `verify_numeric.py`, `verify_timing.py` and `renumbering_proof.py` there (CPU only, about 15 s). The proof (`renumbering_proof.json`) shows every final figure is **byte-identical** to the October output it replaces, so the renumbering changed names only. Shared style helpers are in `../experiments/figure_style.py`; the legacy plotting scripts that wrote the old numbers were removed and stay byte-exact in `../results/full_evaluation_20261015/source.tar.gz`.

The dated October audit folders (`../results/audit_2026101*/`) are kept unchanged as the historical record of how the figures were first checked; each figure folder has a `SUPERSEDED.md` pointing here. Old names in those records map as follows:

| Old file | New file |
|----------|----------|
| fig1_accuracy_equivram | fig1_accuracy_equivram |
| fig7_accuracy_fair | fig2_accuracy_fair |
| fig3_write_savings | fig3_write_savings |
| fig11_footprint | fig4_footprint |
| fig4_decode_scaling | fig5_decode_scaling |
| fig2_latency_comparison | fig6_modeled_traffic_time |
| fig12_energy | fig7_energy |
| fig6_migration_profile | fig8_migration_profile |
| fig14_heatmap | fig9_heatmap |
| fig8_frac_ablation, fig15_frac_compare | removed (never backed by a verified run; in git history) |
| development/ (fig1, fig7, fig14 on consumed development data) | removed (superseded by the full evaluation; in git history) |

Image bytes were not changed by the renaming, so the SHA-256 values in the old deployment records still match the renamed files.

## How to read the figures honestly

- Figure 3 shows only paid versus avoided logical decode writes: 4,959,926 paid vs 52,587,210 avoided = 91.38% of decode writes avoided (hot992). Including the initial cold placement, the whole-request saving is 40.68% pooled and 26.09% equal-task macro ([independent proof](../results/audit_20261016_current_figures/whole_request_writes_hot992_independent_proof.json)). GovReport contributes 77.4% of the avoided rows, so the pooled number depends on the workload. These are logical counters, not physical STT-RAM lifetime measurements.
- Figure 4 separates configured capacity from observed tensor/backing storage at the end of a request. It is not a runtime peak or whole-system memory measurement.
- Figure 5 is measured wall-clock latency. TieredKV is slower (about 75–78 ms/token vs 23–34 ms/token for the others).
- Figures 6 and 7 apply assumed STT-RAM/HBM prices to real event counters. They are not measured device time or energy. Generated lengths differ between methods, so totals alone do not prove per-token efficiency.
- Figure 8 separates deferred candidates from actual transfers.
- Sample error bars are standard errors, not 95% confidence intervals.

Raw evidence: `../results/full_evaluation_20261015/verified_results.json`. Renderer, numeric contracts and independent verifiers: `../results/figures_20261118_renumbered/` (the October originals remain in `../results/audit_20261016_current_figures/` and `../results/audit_20261016_timing/`). Independent re-check scripts used for the presentation: `../docs/presentation/verification/`.
