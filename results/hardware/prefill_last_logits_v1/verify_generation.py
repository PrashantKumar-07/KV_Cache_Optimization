"""All/last prompt-logit parity and frozen ordinary-bulk replay."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
FIELDS = ('prediction', 'generated_token_ids', 'score', 'generation', 'max_gen',
    'termination', 'prompt_sha256', 'state_sha256', 'trajectory_sha256',
    'trajectory_frames', 'counters', 'prefill_query_shapes',
    'explicit_h2d_payload_bytes', 'explicit_d2h_payload_bytes',
    'occupied_host_tensor_bytes', 'reserved_host_tensor_bytes')


def main():
    output = BASE / 'verified_generation.json'
    if output.exists():
        raise FileExistsError(output)
    plan = read(BASE / 'preservation_plan.json')
    implementation = read(BASE / 'implementation.json')
    assert sha(BASE / 'check_generation.py') == plan['driver_sha256'][str((BASE / 'check_generation.py').relative_to(ROOT))]
    assert all(sha(ROOT / path) == value for path, value in implementation['sources'].items())
    prior = ROOT / 'results/hardware/prefill_output_release_v1'
    prior_proof = read(prior / 'verified_generation.json')
    assert prior_proof['verified']
    jobs = read(BASE / 'jobs.json')
    evidence = {}; pairs = replays = tokens = 0
    for geometry, capacities in (('shared', (991, 1090)), ('default', (1024, 2048))):
        name = f'generation_{geometry}.json'
        old = read(prior / name)
        assert sha(prior / name) == prior_proof['evidence_sha256'][name]
        raw = read(BASE / name)
        assert jobs['generation_' + geometry]['status'] == 'finished' and jobs['generation_' + geometry]['exit_code'] == 0
        assert raw['passed'] and not raw['native_checks']
        assert (raw['hot'], raw['cold']) == capacities and len(raw['rows']) == 24
        assert raw['manifest_sha256'] == old['manifest_sha256']
        manifest = ROOT / f'results/hardware/fused_bounded_pool_20261002/generation_{geometry}/generation_manifest.json'
        assert sha(manifest) == raw['manifest_sha256']
        for path, value in raw['source_sha256'].items():
            assert implementation['sources'][path] == value
        frozen = {(r['task'], r['sample_id'], r['seed']): r for r in old['rows'] if r['arm'] == 'fused_bounded'}
        grouped = {}
        for row in raw['rows']:
            key = row['task'], row['sample_id'], row['seed']
            arms = grouped.setdefault(key, {})
            assert row['arm'] not in arms
            arms[row['arm']] = row
            for field in FIELDS:
                assert row[field] == frozen[key][field], (geometry, key, field)
            replays += 1
        assert len(grouped) == 12 and len({key[0] for key in grouped}) == 6
        for key, arms in grouped.items():
            assert set(arms) == {'fused_bounded', 'native_reuse'}
            for field in FIELDS:
                assert arms['fused_bounded'][field] == arms['native_reuse'][field]
            pairs += 1
            tokens += len(arms['native_reuse']['generated_token_ids'])
        evidence[name] = sha(BASE / name)
    assert pairs == 24 and replays == 48
    result = dict(verified=True, archived_current_generator_pairs=pairs,
        frozen_ordinary_bulk_replay_rows=replays, candidate_generated_tokens=tokens,
        equality_fields=list(FIELDS), actual_control='all-prompt-logits generator + ordinary bulk',
        actual_candidate='last-prompt-logits generator + ordinary bulk',
        limits='Reused development documents, six tasks, one model/seed. Exact preservation; no new stream/concurrency, quality-superiority or clean timing proof.',
        evidence_sha256={**evidence, 'preservation_plan.json': sha(BASE / 'preservation_plan.json'),
            'jobs.json': sha(BASE / 'jobs.json')},
        prior_proof_sha256=sha(prior / 'verified_generation.json'), verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
