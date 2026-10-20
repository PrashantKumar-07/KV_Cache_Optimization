import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


design = {
    'recorded_utc': datetime.now(timezone.utc).isoformat(),
    'stage': 'CPU-proven mechanism proposal; no GPU outcomes, source defaults unchanged',
    'primary_outcomes_used_to_tune': False,
    'mechanism': {
        'name': 'Exposure-calibrated actual-query prefill prior',
        'observations': [
            'src/attention.py:102-103 sums probability over all observation queries and query heads.',
            'src/tiered_kv_cache.py:257-258 assigns that sum to cumulative score with age zero.',
            'experiments/longbench_eval.py:1037-1048 adds head-mean, query-mean attention once per decode forward.',
            'src/tiered_kv_cache.py:687 evicts by cumulative score divided by age clamped to one.',
            'src/tiered_kv_cache.py:264 cold last-access is recency metadata, not a prefill cumulative score.',
        ],
        'inference': 'Prompt-hot evidence and subsequent decode evidence use inconsistent head units and exposure counts. This supplies an explicit retention prior strength dependent on model head count and window size.',
        'not_proven': 'The old strength could be beneficial; dimensional inconsistency alone does not prove quality degradation or invalidate preserved measured results.',
        'fixed_change': 'After unchanged actual-Q bifurcation, divide each prompt-hot cumulative prior by captured query-head count and set its age to captured observation-window length. No free strength parameter.',
        'causal_interpretation': 'The common window includes causal future-key zero probabilities. It is a window-average attention prior, not a per-token birth-conditioned attention estimate. Recent-window rows stay protected.',
        'unchanged': ['Initial hot/cold token IDs and K/V bytes', 'Cold recency exposure', 'Promotion/hysteresis', 'Sinks/recent window', 'Capacities and physical storage', 'Generation and scoring protocol'],
        'CPU_proofs': 'test_exposure_prior.py: 10 tests passed; initial payload identity, exact normalization, duplicated-head invariance, analytic uniform causal softmax, short/empty observations, actual multi-query decode averaging, missing-Q rejection.',
        'implementation': 'exposure_prior.py: ExposureCalibratedPriorMixin and ExposureCalibratedPriorPolicy',
    },
    'proposed_development_gate': {
        'approval_status': 'Proposal for root independent review and launch preregistration; not a launched frozen experiment',
        'data': 'Exactly original135 historical development rows. Do not select examples using new fullrun losses. Do not tune on the other1339 consumed rows.',
        'candidate': 'Single exposure-calibrated actual-Q Tiered hot992/cold2048/quota32, frac.75, sinks4/recent128, lambda0, BF16.',
        'controls': 'Reuse fullrun exact matching original135 tiered_hot992 and faithfulSnap1024 records only if generation/runtime/model/input hashes match. Source changes must be isolated and recorded.',
        'primary': 'Equal six-task aggregate paired gain against old Tiered >=1 point, document-cluster95% lower bound >0; no task mean loss >5 points; point gain against faithful Snap >0.',
        'statistics': 'Task-stratified target-document cluster bootstrap10000, seed0, target_document_v2; Qasper/Narrative repeated documents remain clustered.',
        'failure_rule': 'Stop candidate expansion on failure. Do not sweep prior multiplier/window or enlarge sample until CI passes.',
        'resource_next_if_pass': 'Full-model parity of unchanged prefill payload; fixed teacher-forced quiet ABBA timing and CPU+GPU scoped memory before any native adoption.',
    },
    'prospective_independent_reserve': {
        'file': 'fresh_document_reserve.json',
        'sha256': digest(HERE/'fresh_document_reserve.json'),
        'available': {'2wikimqa': {'rows':200,'document_groups':200}, 'musique': {'rows':200,'document_groups':200}, 'qmsum': {'rows':200,'document_groups':35}, 'multi_news': {'rows':200,'document_groups':200}},
        'selection': 'Entire eligible population reserved before tuning. No new development records opened.',
        'use_rule': 'Only after candidate passes original135 development and separate resource gates; freeze candidate and comparator protocol before any generation.',
        'limitations': 'Exact casefold/whitespace whole-context and row-ID exclusion across all dataset variants and recorded results/manifests. Unrecorded manual outcomes, semantic near duplicates and partial Wikipedia overlap unverified. New tasks imply a new transfer-validation claim, not a new independent replication of old six-task performance.',
    },
    'rejected_directions': [
        {'direction':'Per-KV-head retention', 'reason':'Already implemented grouped_retention.py and failed preregistered six-row gate. Do not resurrect as new head-allocation experiment under another name.', 'evidence':'results/audit_20261013/reviewer/grouped_pilot_verified.json'},
        {'direction':'Current-query cold selection combination', 'reason':'Failed registered lower confidence bound gate. Do not expand to seek significance.', 'evidence':'results/audit_20261015_query_combination/verified_quality.json'},
        {'direction':'Head-max importance', 'reason':'Existing historical broader-development screen already exists; do not duplicate without reading its failure and novelty evidence.', 'evidence':'results/baseline_fidelity/head_max_broader_dev_20261002/verified_report.json'},
    ],
    'current_primary_literature_review': [
        {'url':'https://github.com/yuzhenmao/IceCache', 'checked':'2026-10-16', 'relevance':'Current CPU-offload comparator: dynamic M-DCI query-based page retrieval. Repository supports Mistral and separate BF16 branch. It recommends >=64 CPU threads; installation uses Torch2.4, FlashAttn2.6.1 and M-DCI/OpenBLAS. Isolated baseline environment is appropriate; do not mutate frozen current runtime.'},
        {'url':'https://github.com/FFY0/AdaKV', 'checked':'2026-10-16', 'relevance':'Head-wise adaptive cache allocation is established prior work. A head-specific residency variant cannot be claimed as new just because it preserves eight GQA KV heads.'},
        {'url':'https://github.com/Zefan-Cai/KVCache-Factory', 'checked':'2026-10-16', 'relevance':'Current unified comparison repository explicitly supports KV-head granularity for multiple methods, including SnapKV. Comparing only to legacy expanded32-head Snap layout does not establish a memory SOTA win. Validate GQA-efficient baseline before any such claim.'},
        {'url':'https://arxiv.org/abs/2306.14048', 'checked':'2026-10-16', 'relevance':'Accumulated historical attention retention is existing H2O prior art. Calibrating its evidence units is an engineering hypothesis, not a new fundamental algorithm by itself.'},
    ],
    'claim_boundary': 'A passing calibration screen supports corrected retention quality on development only. SOTA superiority needs independent data, faithful current comparators, measured matched resource envelope and finite preregistered objective. A system Pareto claim may concern accuracy versus memory/latency/logical writes, but simulated STT device energy/endurance is not a measured hardware win.',
    'initial_inventory_attempt': 'First inventory attempt failed before output: Python splitlines split an embedded Unicode line separator in a valid JSON string. Fixed to literal newline JSONL parsing; no candidate outcomes were run.',
    'files_sha256': {str(path.relative_to(ROOT)):digest(path) for path in [HERE/'exposure_prior.py', HERE/'test_exposure_prior.py', HERE/'cpu_tests.log', ROOT/'src/attention.py', ROOT/'src/tiered_kv_cache.py', ROOT/'experiments/longbench_eval.py']},
}
target=HERE/'design.json'
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(design, indent=2)+'\n')
print(target)
