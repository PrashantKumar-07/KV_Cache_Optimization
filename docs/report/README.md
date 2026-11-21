# Thesis report (TieredKV, 2-tier version)

`../../Thesis_report.pdf` is the written report (26 A4 pages). Its cover carries the same information as the presentation title slide (plain white page). Text uses black and one blue accent only. It covers background, related work (including the FlexiCache overlap), the design, methodology, results, verification, limitations, future work and references.

- Source: `report.html` (figures are loaded from `../../figures_final/` and `../../figures/`, logo from `../presentation/assets/`).
- Build: `python3 build.py` (needs `google-chrome` and poppler-utils). It renders twice so the table of contents gets real page numbers, then writes `Thesis_report.pdf` in this folder; move it to the repository root (only the root copy is tracked).
- Every number comes from verified sources: `results/figures_20261118_renumbered/task_values.csv`, `timing_values.csv` and `endpoint_values.csv`, plus `results/gqa_baseline_20261016/verified_results.json`. Figure numbers 1–9 match the presentation; the architecture diagram is Figure A.
