# Native pool current-query selector evidence

Research module: `experiments/pool_query_selector.py`. Ordinary CPU CLI remains
restricted to LRU. Selection scans live cold K/V through paired native reads,
uses the existing rotary query/softmax/top-k rule, and preserves exposed width.

1. [Original preregistration](plan.json) and archived ordinary control freeze
   hot991/cold1090/quota32 before implementation.
2. [Correction plan](correction_plan.json) preserves the initial failed CUDA
   run. CPU shadow metadata needed live-row indices moved onto the query device.
   Original failed source/raw/job remain available; corrected source is separate.
3. [Correctness proof](verified_correctness.json): four tiny pairs,266 tests,
   three full-model GPU implementations exact across12 predictions/13 cache
   frames/384 visible-KV and choice observations. Native selector canonical
   CPU reads zero. Its1.379GB H2D over12 steps is charged.
4. [Fresh resource proof](verified_resource.json):16K50 passes its declared
   576MiB combined active ceiling at544.321MiB. Owned rounded host256MiB,
   reserved5089.492MiB and prefill18.62783GB. All teacher predictions match;
   owned roots/hooks release. H2D5.826GB versus0.316GB ordinary LRU.
5. [Development preregistration](development_plan.json):12 reused requests,
   six tasks, one seed, paired current native-last-logit ordinary control and
   query selector. Contexts disjoint from the100-document final holdout.
   Nonnegative task-balanced direction passes(+2.132points); this does not prove independent quality.

6. [Timing stop decision](verified_timing.json):3ABBA16K50,one warmup/row.
   Median58.684→139.831ms(2.383×), p9564.720→143.339ms(2.215×), both fail
   the2%regression guard. TTFT passes. Stop wider quality/timing/adoption.
   All teacher predictions replay; ownhooks release;25before/afterGPU0 compute
   PID observations are quiet. No continuous utilization assurance.
   [Diagnostic figure](query_tradeoff.pdf), [data](query_tradeoff.csv).

The verification scripts check archived/current source hashes, terminal job
records, raw outputs and cleanup before writing a new proof. Existing proof
files cannot be overwritten. Generation observer wall times cannot support
latency claims. Quiet3ABBA timing and independent matched-resource quality
remain required before ordinary adoption. Complete512MiB, exact faithful
baseline allocation/attention matching and physicalSTT remain unproved.

See [project status](../../../PROJECT_STATUS.md) for the current decision.
