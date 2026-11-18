# Thesis presentation (2-tier TieredKV)

`../../Thesis_presentation_2tier.pdf` is the presentation deck (27 slides, same template as the earlier mid-term deck, no footer).
`QUICK_SUMMARY.md` (and its PDF) is the talk script: what to say per slide group, a simple walkthrough of the architecture diagram, likely questions, and what changed from the old 3-tier Quick Summary.

- Slides 1–2 and 11–27 are built from `new_slides.html` in this folder (slide 12, "Why STT-RAM", uses the assumed prices in `src/cost_model.py`).
- Slides 3–10 are reused from the earlier deck (`Thesis ppt.pdf`, pages 3–10) with only the footer text covered; they contain the background material (LLM inference, KV cache, StreamingLLM, SnapKV, Quest, gap analysis).
- The architecture diagram is `../../figures/tieredkv_architecture_v2.svg` (PDF/PNG alongside). It is drawn from the executed code path (`experiments/longbench_eval.py::TieredKVPolicy` and `src/tiered_kv_cache.py` in the frozen source of `results/full_evaluation_20261015/source.tar.gz`).

## Rebuild

```bash
cd docs/presentation
google-chrome --headless=new --no-sandbox --virtual-time-budget=15000 --no-pdf-header-footer \
  --print-to-pdf=new.pdf "file://$PWD/new_slides.html"
pdfseparate new.pdf n-%02d.pdf
pdfseparate "../../Thesis ppt.pdf" o-%02d.pdf
# cover the old "IIT JODHPUR · M.TECH SUMMER THESIS" footer on pages 3-10 (needs PyMuPDF)
python3 -c "import pymupdf
for i in range(3, 11):
    d = pymupdf.open(f'o-{i:02d}.pdf'); p = d[0]
    p.add_redact_annot(pymupdf.Rect(48, 503, 262, 520), fill=(245/255, 248/255, 251/255))
    p.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE); d.save(f'c-{i:02d}.pdf')"
pdfunite n-01.pdf n-02.pdf c-0{3..9}.pdf c-10.pdf n-{03..19}.pdf ../../Thesis_presentation_2tier.pdf
# talk script
python3 md2html.py QUICK_SUMMARY.md qs.html
google-chrome --headless=new --no-sandbox --allow-file-access-from-files --virtual-time-budget=8000 --no-pdf-header-footer \
  --print-to-pdf=QUICK_SUMMARY.pdf "file://$PWD/qs.html"
```

## How the numbers were verified (18 Nov 2026)

Every number on the slides was checked against raw outputs, independently of the code that drew the figures:

1. `verification/independent_check.py` re-scores all 10,318 saved predictions of `results/full_evaluation_20261015` with its own QA-F1 and ROUGE-L, decodes every stored token-ID list with the real tokenizer, and re-sums the write counters. Result: 0 score mismatches, 0 text mismatches, all 42 task × method values equal to the plotted table. The same re-scoring of the 1,474 official GQA SnapKV records gives a task average of 39.523.
2. `verification/repro_driver.py` re-executes 18 recorded requests (6 tasks × full / SnapKV-adapted / TieredKV hot992) on an L40S with the frozen evaluation source. All 18 reproduce identical generated tokens, scores and write counters.
3. All 30 Fig. 5 latency medians were recomputed from the raw per-step timings in `results/audit_20261016_timing/timing.json`: all match.
4. Fairness audit of the raw records: identical tokenized input for all 7 methods on every row; no output exceeds its task's token limit; gold answers are only read after generation (for scoring); TieredKV hot992 never attended more than 1,025 rows in a decode step (same as the 1,024-row baselines), measured from the KV tensors given to the model.
5. Whole-request write avoidance recomputed from post-prefill cold occupancy plus decode counters: 40.6793 % pooled, 26.0881 % task average.
6. `pytest`: 601 passed, 1 skipped (CUDA-only); `scripts/smoke.py` passes.

The old Figures 8 and 15 had no verified source run and were removed. The final figures are renumbered 1–9 in slide order; see `figures_final/README.md` for the old → new map.

Run the scripts from the repository root, for example:
`python docs/presentation/verification/independent_check.py <repo-root>` (needs the model tokenizer at `../models/mistral-7b-instruct`) and
`python docs/presentation/verification/repro_driver.py <extracted source.tar.gz root> <repo-root>` (needs a GPU).
