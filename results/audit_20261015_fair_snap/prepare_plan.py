"""CPU-only prospective registration; refuses to replace an existing plan."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments'))
from provenance import atomic_json, model_artifact_identity, runtime_identity


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


if __name__ == '__main__':
    assert not (BASE / 'plan.json').exists(), 'Keep the prospective registration immutable'
    snapbase = ROOT / 'results/audit_20261013/snap_projection_reuse_gpu'
    nativebase = ROOT / 'results/audit_20261013/reusable_root_parity'
    snap = json.loads((snapbase / 'plan.json').read_text())
    native = json.loads((nativebase / 'plan.json').read_text())
    oldtiming = json.loads((ROOT / 'results/audit_20261013/reusable_root_resources/plan.json').read_text())
    assert all(sha(ROOT / name) == value for name, value in snap['source_sha256'].items())
    assert len(snap['source_sha256']) == 5918
    assert sha(ROOT / snap['source_archive']) == snap['source_archive_sha256']
    bridge = {name: {'prior': value, 'current': snap['source_sha256'].get(name)}
              for name, value in native['source_sha256'].items()
              if snap['source_sha256'].get(name) != value}
    # This bridge limits reuse of the older native proof to unchanged native code.
    assert set(bridge) <= {'experiments/snapkv_adapter.py', 'tests/test_hook_registration_rollback.py'}, bridge
    model = model_artifact_identity(snap['model'])
    assert model == snap['model_identity']
    tokenizer = AutoTokenizer.from_pretrained(snap['model'])
    ids = tokenizer((ROOT / snap['prompt']).read_text(), truncation=False, return_tensors='pt').input_ids
    assert ids.shape == (1, 30907)
    contexts = [8192, 16384, 30000]
    native_raw = json.loads((nativebase / 'parity.json').read_text())
    with np.load(snapbase / 'reused_logits.npz') as arrays:
        snap_predictions = [int(arrays[str(step)].reshape(-1).argmax()) for step in range(1, 51)]
    references = {'16384': {'snap': snap_predictions,
                  'native': native_raw['pairs']['16384']['candidate']['predicted_token_ids']},
                  '8192': {'native': native_raw['pairs']['8192']['candidate']['predicted_token_ids']}}
    prerequisites = [snapbase / name for name in ('plan.json', 'verified_parity.json', 'parity.json', 'reused_logits.npz')]
    prerequisites += [nativebase / name for name in ('plan.json', 'verified_parity.json', 'parity.json')]
    prerequisites += [ROOT / 'results/audit_20261013/reusable_root_resources/verified_resources.json',
                      ROOT / 'results/audit_20261013/snap_projection_reuse_full_suite.log']
    for base in (snapbase, nativebase):
        proof = json.loads((base / 'verified_parity.json').read_text())
        assert proof['verified'] is True
        prerequisites.extend(ROOT / name for name in proof['evidence_sha256'])
    plan = {
        'registered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'prospective; root review required before GPU launch',
        'hypothesis': 'At fixed scoped budgets, reusable native actual-query retention may improve full-loop latency over faithful expanded-head SnapKV without duplicate prefill projections. Failure is valid evidence; no tuning follows this run.',
        'arms': {
            'snap': {'implementation': 'SnapKVAdapter', 'capacity': 904, 'window': 32, 'kernel': 5,
                     'reuse_projection': True, 'heads': 'upstream expanded query heads', 'decode': 'growing cache'},
            'native': {'implementation': 'ReusableExactPolicy + PromptQueryCapture + SharedAppendLease',
                       'hot': 1024, 'cold': 2048, 'fraction': .75, 'quota': 32, 'sink': 4, 'recent': 128,
                       'actual_Q_prefill': True, 'persistent_exact_pinned_root_bytes': 276824064}},
        'contexts': contexts, 'order': ['snap', 'native', 'native', 'snap'] * 3,
        'settings': {'device': 'physical GPU0', 'seed': 0, 'dtype': 'BF16', 'prefill': 'sdpa',
                     'decode': 'eager', 'decode_steps': 50, 'teacher_tokens': 51,
                     'last_logits': True, 'warmup_per_row': 1, 'measured_repeats_per_row': 2,
                     'blocks_per_context': 3, 'rows_per_context': 12,
                     'measured_forwards_per_arm_per_context': 600,
                     'context_order': '8192 then16384 then30000; no adaptive ordering',
                     'finite_check': 'Every complete forward vocabulary array checked on GPU; one scalar finite flag delivered to host inside both arms full-loop timing'},
        'timing_ratio_guards': {'median_itl_ms': .95, 'p95_itl_ms': 1.05, 'median_ttft_ms': 1.10},
        'advancement': 'Require all three timing guards and both scoped512MiB guards at EVERY context; also require pooled timing guards. Pooled600x3 steps/arm is descriptive fixed equal-context mixture, not a general workload estimate. Preserve failure; no capacity/prompt/stream tuning after results.',
        'scoped_budget_bytes': 512 * 2**20,
        'scope_formula': {
            'native': 'decode peak GPU allocated minus deduplicated static model storage + observed pinned allocator block + canonical CPU arrays + CPU streak array peak + CPU address array peak +2*(bulk_descriptor_h2d_bytes+cold_boundary_descriptor_h2d_bytes) of same final50-step trial',
            'snap': 'decode peak GPU allocated minus deduplicated static model storage; no native root charged logically to Snap',
            'idle': 'Count native264MiB root active and idle; separately record experiment-retained root during Snap rows, which is outside Snap logical ownership',
            'descriptor_reservation': '2x cumulative descriptor transfer bytes per trial is inherited conservative reservation, not simultaneous measured occupancy',
            'excluded': 'Python/native container rounding, hidden cached host/driver allocations, total RSS and whole-machine ownership; GPU reserved reported separately, not substituted for allocated scope'},
        'diagnostic': {'context': 16384, 'order': ['snap', 'native'], 'requests_per_arm': 1,
                       'boundaries_per_arm': 51, 'snapshots': 'After caller apply_tuple_kv_inplace for native and after forward for Snap; forward out and temporary tuple references expired before snapshot',
                       'timing': 'Copies, synchronization and storage inspection invalidate ALL diagnostic timing; excluded from timed rows and guards',
                       'startup': 'Ordered first observed instrumented requests and native provisioning observation only; no causal coldTTFT comparison'},
        'expected_native_leases': 55, 'expected_allocation_release_pairs': 1,
        'expected_final_live_roots': 0, 'expected_quiet_observations': 77,
        'correctness': 'Both16K predictions must match corresponding frozen fullmodel proof; native8K also priorproof. All warm/measured/diagnostic arm/context predictions including prefill repeat exactly and every full-logit array finite. Snap8K/both30K have no preexisting full-vocabulary/KV parity proof; broader correctness remains open. Different arms need not predict identically.',
        'reference_predictions': references,
        'reference_scope': {'snap': ['16384'], 'native': ['8192', '16384'],
                            'missing_full_logit_KV_proofs': ['snap8192', 'snap30000', 'native30000']},
        'full_prompt_tokens': 30907,
        'input_ids_sha256': {str(ctx): hashlib.sha256(ids[:, :ctx].numpy().tobytes()).hexdigest() for ctx in contexts},
        **{name: snap[name] for name in ('model', 'prompt', 'prompt_sha256', 'teacher_stream', 'teacher_stream_sha256',
                                       'source_archive', 'source_archive_sha256', 'source_sha256')},
        'model_identity': model, 'runtime': runtime_identity(),
        'quiet_helper': oldtiming['quiet_helper'], 'quiet_helper_sha256': oldtiming['quiet_helper_sha256'],
        'prerequisite_sha256': {str(path.relative_to(ROOT)): sha(path) for path in set(prerequisites)},
        'prior_native_proof_source_bridge': {'changed_prior_files': bridge,
             'current_added_files': sorted(set(snap['source_sha256']) - set(native['source_sha256'])),
             'interpretation': 'All native source unchanged from prior native16K/8K fullmodel proof; only Snap and its tests changed'},
        'runner_sha256': sha(BASE / 'run_benchmark.py'), 'registration_script_sha256': sha(__file__),
        'launch_command': 'CUDA_VISIBLE_DEVICES=0 ../venv_kvcache/bin/python -u results/audit_20261015_fair_snap/run_benchmark.py > results/audit_20261015_fair_snap/run.log 2>&1',
        'limits': ['One Mistral local snapshot, one real text prefix and same consumed fixed teacher stream; no free-generation or independent quality claim',
                   'Three context prefixes are correlated; no independent corpus/context generalization',
                   'Faithful upstream selection/storage semantics, optin native-Q projection reuse exact16K proof; Snap does not request unused attention weights, native requires feedback',
                   'Model and root preloaded for steady rows; constructor/reset/hook setup outside per-trial TTFT in both arms',
                   'Quiet GPU process checks bracket requests, not continuous host/driver/activity isolation',
                   'Analytical device/STT costs excluded; pinned CPU DRAM surrogate, no physical STT claim',
                   'Fixed capacities satisfy independent budget guards if observed; not exact whole-resource equality, no quality superiority or default-adoption decision']}
    atomic_json(BASE / 'plan.json', plan)
    print(json.dumps({'registered': True, 'source_files': len(plan['source_sha256']),
                      'contexts': contexts, 'expected_native_leases': 55,
                      'prior_source_bridge': plan['prior_native_proof_source_bridge']}, indent=2))
