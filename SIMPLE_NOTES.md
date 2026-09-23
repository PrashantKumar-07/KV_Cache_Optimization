# TieredKV — Simple Notes (for future me)

Plain-English record of what this project is, how it works, and every big
doubt I raised with its answer. Numbers here match the committed results.

## 1. The project in one paragraph

LLM inference stores past keys/values (KV cache) that grows with context
length and can exhaust GPU memory. Standard fix: keep only ~1024 tokens
and **throw away the rest forever** (StreamingLLM, H2O, SnapKV). My idea:
instead of throwing demoted tokens away, park them in a **cheap second
tier (STT-RAM)** where they stay searchable and can be **promoted back**
when the query needs them. That is the whole paper: a victim cache for
KV eviction, plus the write savings that fall out of it.

## 2. The tiers (what lives where)

| Tier | Tokens (headline) | Size bf16 | Physical home today |
|------|-------------------|-----------|---------------------|
| 1. GPU VRAM (hot working set) | 1024 | ~128 MB | GPU VRAM |
| 2. STT-RAM (warm victim cache) | 2048 | ~256 MB | GPU VRAM too (simulated!) |
| 3. DRAM (cold) | 0, never used | — | does not exist in any run |

Math: Mistral-7B needs 32 layers × 8 heads × 128 dim × 2 (K,V) × 2 bytes
= **128 KiB per token**. So 1024 ≈ 128 MB, 2048 ≈ 256 MB, total ≈ 384 MB
(Fig 11 shows 403 MB with overhead; baselines show 134 MB).

Important honesty points:
- **STT-RAM is simulated.** Its tensors sit beside the VRAM tensors in GPU
  memory. Only its *price* (read/write latency + energy) is modeled, from
  published device parameters ("tentpole"). The policy, the attention, the
  GPU timing, and the scores are all real.
- **No DRAM tier exists.** An early revision carried DRAM tensors with no
  fetch-back path; that dead code has been removed outright. Dropped
  tokens are discarded, full stop — this is a **2-tier + drop** system.

## 3. How a prompt flows through the system

1. **Bifurcation (prefill):** rank prompt tokens by SnapKV importance. Top
   ~1024 (plus pinned sinks/window) → VRAM. Next chunk → STT (up to a cap).
   Rest → dropped forever.
2. **Decode loop:** each step attends over VRAM + at most 32 exposed STT
   tokens. Sketch check finds STT pages matching the current query.
3. **Promotion:** hot STT pages move up to VRAM ("resurrection" — the
   opposite of baselines' permanent eviction).
4. **Demotion:** cold VRAM tokens move down to STT. If a shadow backup of
   that token already sits in STT, re-demotion **reactivates the backup at
   zero write cost** — that is the write saving.

## 4. What `frac` means

`frac` = fraction of STT filled at prefill (`sttram_bifurcation_frac`).
- `frac=1.0` → STT starts 100% full (2048/2048). Every later demotion
  overflows STT and destroys a backup → write savings ≈ 0%.
- `frac=0.75` → STT starts 75% full (1536/2048). 512 empty rows of
  headroom let demote→promote→re-demote cycles hit backups → ~80% savings.
- The missing 25% of tokens are **dropped, not moved to DRAM**.

Sweep result (5 short tasks): F1 37.92 / 38.14 / 38.51 / 38.67 at
frac 0.25 / 0.5 / 0.75 / 1.0, while write savings jump from ~0–22% (1.0)
to ~80–88% (≤0.75). Accuracy barely moves because `frac` never touches
what the model attends to — only which *cold* tokens are retained.

## 5. Why the budgets are small (1024) even on a 45 GB GPU

- It is an experimental **control**, not a capacity claim: equal silicon
  for every method, so we compare policies, not memory sizes.
- Unbounded budgets erase all differences (everyone = Full Cache).
- Real deployments still hit the wall: 70B models ≈ 2.6 MB/token
  (32k ctx ≈ 83 GB); batching multiplies everything; 128k ctx on 7B =
  16 GB per request.
- The whole literature (StreamingLLM/H2O/SnapKV) evaluates this way;
  reviewers require the same protocol.

## 6. Is 1024+2048-vs-1024 fair? Yes, with two sentences

1. **Equal VRAM silicon** (1024 tokens of precious GPU memory). STT is the
   intervention being tested, like adding cheap flash next to DRAM.
2. **Capped exposure** (32): extra residents can't buy extra FLOPs. Plus a
   second control run exists: **992 VRAM + 32 exposed = 1024 attended**,
   exactly matching baselines (`final_equal_attended.json`).

## 7. Headline numbers (equal-VRAM, 6 LongBench tasks)

Avg F1: Full 43.89, StreamingLLM 34.08, H2O 35.63, SnapKV 35.86,
**TieredKV 36.58**. Decode ITL@16k: 61.7 / 24.7 / 24.8 / 24.8 / **25.1** ms.
Full-KV is 2.5× slower for +7.3 F1. All accuracy+latency numbers are
measured; only STT energy/latency *prices* are modeled.

## 8. My doubts, answered short

- **"Project is fake?"** No: real model, real attention-driven policy, real
  GPU timing, real scores. Simulated: STT device physics only. Standard
  systems methodology — but every number must stay labeled measured vs
  modeled.
- **"Evicted-then-needed tokens cause failures?"** Yes for baselines
  (permanent drop). TieredKV's answer is promotion-as-resurrection within
  STT capacity; the dropped tail costs ≤0.8 F1 (frac sweep).
- **"Why not keep everything in CPU DRAM like FlexGen?"** Graveyard mode
  changes zero numbers; fetchable mode breaks equal-budget fairness and is
  a second paper, not a patch.
- **"Headline frac?"** 0.75 (mechanism visibly works); show 1.0 in ablation
  to prove the tradeoff.
- **"Retired figures?"** Old Fig 4 scatter (5 dots, in master table),
  Fig 9 (all bars "40"), Fig 10 (5 identical lines), Fig 13 (Fig 3 with
  less detail). Facts preserved as text + master table.

## 9. Still to do (from the plan in chat)

- [x] Dropped-tail A/B rerun — DONE without new GPU time: `e2_frac1_0`
  vs `e2_frac0_75` are the same batch, proven bit-identical to final code
  (all 5 short tasks). ΔF1: −1.94 … +0.94 per task (≈0.0 mean with gov);
  write savings 0–22% → 78–92%. Multfldqa pays most (−1.94) for the tail.
- [x] Headline switched to `frac=0.75`
  (`results/final_equal_vram_frac075.json`, provenance inside); README
  table + Fig 1 + Fig 6 + Fig 12 + Fig 14 + master table regenerated.
- [x] 2-tier framing, STT capacity statement (2048 tok / 256 MiB),
  fairness lines, resurrection framing — in README + explanation doc.
- [x] Tentpole sensitivity run (migration 0.803/0.346/0.180 s;
  energy invariant 70.63 J) — recorded in README.
- [x] Fair-config (992+32) TieredKV @ 0.75 rerun — DONE on GPU
  (`results/fair075_short/gov.json` → merged
  `results/final_equal_attended_frac075.json` with provenance). Result:
  35.46 vs SnapKV 35.86 (−0.40, inside SEs — a statistical tie, deficit
  concentrated in hotpotqa −4.06). Fig 7 / Fig 2 / Fig 3 / Tables 2–3
  regenerated; README + doc verdicts updated (no more "tie" claim).
- [ ] Optional: attention-mass-on-dropped-tokens analysis (needs new runs).
