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

The frozen audit records under `../results/audit_2026101*/` (renderers, deployment hashes, numeric contracts) still use the old file names. They are not edited, because they are evidence. The plotting scripts `experiments/plot_figures_v2.py` and `experiments/plot_publication_figures.py` are also hash-pinned by that provenance (and imported by the frozen renderer), so they keep the old output names too; `experiments/plot_frac_ablation.py` and `plot_frac_compare.py` produced the removed unverified figures. If you re-render, rename the outputs with this table:

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

Raw evidence: `../results/full_evaluation_20261015/verified_results.json`. Renderer, numeric contracts and independent verifier: `../results/audit_20261016_current_figures/` and `../results/audit_20261016_timing/`. Independent re-check scripts used for the presentation: `../docs/presentation/verification/`.
