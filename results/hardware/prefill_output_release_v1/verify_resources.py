"""Read-only caller lifetime/preservation and unchanged phase-memory gate."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
spec = importlib.util.spec_from_file_location('snapshot_verifier',
    ROOT / 'results/hardware/bulk_resource_audit_v1/verify_results.py')
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


def main():
    output = BASE / 'verified_resources.json'
    if output.exists():
        raise FileExistsError(output)
    plan = read(BASE / 'implementation.json')
    assert sha(BASE / 'source.tar.gz') == plan['source_archive_sha256']
    with tarfile.open(BASE / 'source.tar.gz') as archive:
        sources = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
            for m in archive.getmembers() if m.isfile()}
    assert sources == plan['sources_after']
    assert all(sha(ROOT / p) == value for p, value in sources.items())
    # Prove exactly one deletion in each caller, preserving every other AST node.
    for path, digest in plan['sources_before'].items():
        before = BASE / (Path(path).name + '.before')
        assert sha(before) == digest
        old, new = ast.parse(before.read_text()), ast.parse((ROOT / path).read_text())
        function_name = 'main' if 'measure_fresh' in path else 'generate_with_policy'
        function = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
        deletions = [n for n in ast.walk(function) if isinstance(n, ast.Delete)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'out']
        assert len(deletions) == 1
        parents = [value for n in ast.walk(function) for _, value in ast.iter_fields(n)
            if isinstance(value, list) and deletions[0] in value]
        assert len(parents) == 1
        parents[0].remove(deletions[0])
        assert ast.dump(old) == ast.dump(new)
    prior = ROOT / 'results/hardware/fresh_prefill_owner_v1'
    assert sha(prior / 'verified_report.json') == plan['prior_fresh_proof_sha256']
    old = read(prior / 'ordinary.json')
    jobs = read(BASE / 'jobs.json')
    rows = {}; raw = {}
    fields = ('predicted_token_ids', 'teacher_token_ids', 'input_ids_sha256', 'prompt_sha256',
        'teacher_source_sha256', 'state_sha256', 'counters', 'occupied_host_bytes', 'reserved_host_bytes')
    for arm in ('retained', 'released'):
        d = raw[arm] = read(BASE / (arm + '.json'))
        assert jobs[arm]['status'] == 'finished' and jobs[arm]['exit_code'] == 0
        assert d['passed'] and d['caller_output_lifetime'] == arm
        assert d['fresh_owner_arm'] == 'ordinary' and d['owner_hooks_removed']
        assert d['owner_counts'] == dict(released_forwards=0, restored_forwards=0, released_layers=0)
        assert d['release_driver_sha256'] == sources[str((BASE / 'audit_release.py').relative_to(ROOT))]
        assert d['archived_caller_sha256'] == plan['sources_before']['experiments/measure_fresh_process_memory.py']
        for key in ('source_sha256', 'actual_source_sha256'):
            assert all(sources[p] == value for p, value in d[key].items())
        for key in fields:
            assert d[key] == old[key], (arm, key)
        assert len(d['predicted_token_ids']) == 50 and (d['hot'], d['cold']) == (991, 1090)
        assert d['reserved_host_bytes'] == d['owned_pinned_peak_bytes'] == 144.25 * 2**20
        assert d['owned_host_allocator_block_peak_bytes'] == 256 * 2**20
        assert d['owned_pinned_after_cleanup_bytes'] == d['owned_host_allocator_block_after_cleanup_bytes'] == 0
        model = d['resource_model']; base = model['constant_model_storage_bytes']
        assert sum(model['model_ranges'].values()) == base == 14483464704
        for name in ('loaded_model_snapshot', 'decode_final_snapshot'):
            snapshot.check_snapshot(model[name], model['model_ranges'])
        calls = d['logits_lifetime_rows']
        assert len(calls) == 51 and calls[0]['logits_bytes'] == 1000 * 2**20
        assert calls[0]['shape'] == [1, 16384, 32000]
        assert calls[1]['prefill_logits_alive_before_forward'] == (arm == 'retained')
        assert all(not r['prefill_logits_alive_before_forward'] for r in [calls[0], *calls[2:]])
        peaks = d['gpu_phase_peaks']; host = d['owned_host_allocator_block_peak_bytes']
        rows[arm] = dict(active_decode_envelope_mib=(peaks['decode_peak'] - base + host) / 2**20,
            reserved_envelope_mib=(peaks['whole_reserved_peak'] - base + host) / 2**20,
            prefill_peak_gb=peaks['prefill_peak'] / 1e9)
    assert raw['retained']['prototype_counts'] == raw['released']['prototype_counts'] == old['prototype_counts']
    assert raw['retained']['gpu_phase_peaks'] == old['gpu_phase_peaks']
    for key in ('prefill_peak', 'whole_allocated_peak', 'whole_reserved_peak'):
        assert raw['retained']['gpu_phase_peaks'][key] == raw['released']['gpu_phase_peaks'][key]
    reduction = 100 * (1 - rows['released']['active_decode_envelope_mib'] / rows['retained']['active_decode_envelope_mib'])
    result = dict(verified=True, output_lifetime_gate_passed=True,
        phase_memory_primary_passed=reduction >= 10, reduction_pct=reduction, rows=rows,
        frozen_replay_rows=2, all_predictions_state_counters_and_transfers_match=True,
        complete512MiB_gate_passed=False, speed_claim=False,
        decision='Keep lifetime bug fix; stop expansion as a 10% phase-memory optimization.',
        phase_scope='Unchanged original phase starts at compressed prefill BEFORE explicit release. Its high-water mark includes the live-logits handoff; no reset moved to improve this gate.',
        evidence_sha256={p: sha(BASE / p) for p in ('plan.json', 'implementation.json', 'source.tar.gz',
            'jobs.json', 'retained.json', 'released.json', 'tests_before.log', 'tests_after.log')},
        verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
