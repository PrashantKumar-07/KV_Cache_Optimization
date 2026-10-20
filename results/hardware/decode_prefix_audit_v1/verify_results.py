"""Audit prefix facts, frozen replay and historical live-storage attribution."""
import collections
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def main():
    output = BASE / 'verified_report.json'
    if output.exists(): raise FileExistsError(output)
    plan = read(BASE / 'plan.json')
    cli = ROOT / 'results/hardware/promotion_planning_screen_20261002/cli/verified_report.json'
    prior_dir = ROOT / 'results/hardware/reuse_visible_model_20261002'
    assert sha(cli) == plan['frozen_control_proof_sha256']
    assert sha(prior_dir / 'verified_parity_resource.json') == plan['prior_trace_proof_sha256']
    assert sha(BASE / 'source.tar.gz') == plan['source_bundle_sha256']
    with tarfile.open(BASE / 'source.tar.gz') as archive:
        sources = {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                   for m in archive.getmembers() if m.isfile()}
    assert all(sha(ROOT / p) == value for p,value in sources.items())
    jobs = read(BASE / 'jobs.json')
    assert len(jobs) == 2 and all(j['status'] == 'finished' and j['exit_code'] == 0 for j in jobs)
    results = {}; replayed = 0
    fields = ('state_sha256','history_sha256','every_visible_state_sha256',
              'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes',
              'occupied_host_bytes','reserved_host_bytes','native_d2h_calls','native_h2d_calls','per_layer_initial_d2h_calls')
    trial_fields = ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps')
    for name,hot,cold in (('shared',991,1090),('default',1024,2048)):
        raw = read(BASE / (name+'.json'))
        assert raw['passed'] and raw['prefix_audit'] and len(raw['rows']) == 8
        assert raw['prefix_audit_sha256'] == sources['experiments/audit_decode_prefix.py']
        assert all(sources[p] == value for p,value in raw['source_sha256'].items())
        assert (raw['config']['hot'],raw['config']['cold']) == (hot,cold)
        assert raw['config']['contexts'] == [4096,8192,16384,31500]
        assert raw['config']['decode'] == 12 and raw['config']['blocks'] == raw['config']['warmup'] == 0
        older_path = (prior_dir / 'parity_shared.json' if name == 'shared' else
                      ROOT / 'results/hardware/promotion_planning_screen_20261002/model_parity.json')
        older_proof = read(prior_dir / 'verified_parity_resource.json' if name == 'shared' else
                           older_path.parent / 'verified_model_parity.json')
        assert sha(older_path) == older_proof['evidence_sha256'][older_path.name]
        older = read(older_path)
        previous = {(r['context'],r['arm']):r for r in older['rows']}
        for index,context in enumerate(raw['config']['contexts']):
            control,candidate = raw['rows'][index*2:index*2+2]
            assert control['arm'] == 'fused_bounded' and candidate['arm'] == 'native_reuse'
            assert control['context'] == candidate['context'] == context
            for field in fields: assert control[field] == candidate[field], (name,context,field)
            assert len(candidate['every_visible_state_sha256']) == 13
            for row in (control,candidate):
                assert len(row['trials']) == 1 and len(row['trials'][0]['predicted_token_ids']) == 12
                assert row['reserved_host_bytes'] == (cold+64)*131072
                for field in trial_fields: assert control['trials'][0][field] == candidate['trials'][0][field]
                old = previous[(context,'native_reuse' if name == 'shared' else row['arm'])]
                for field in fields: assert row[field] == old[field], ('frozen replay',name,context,field)
                for field in trial_fields: assert row['trials'][0][field] == old['trials'][0][field]
                replayed += 1
        observations = raw['prefix_observations']
        assert len(observations) == 1536
        expected = {(c,s,l) for c in raw['config']['contexts'] for s in range(1,13) for l in range(32)}
        assert {(r['prompt_tokens'],r['step'],r['layer']) for r in observations} == expected
        for row in observations:
            assert row['prefix_bits_equal'] and row['storage_distinct'] and row['capacity_fits'] and row['new_tail_unchanged']
            assert row['hot_before'] == hot and row['new_rows'] == 1
            assert row['raw_rows'] == hot+33 and row['extended_rows'] == hot+row['promoted_rows']+1
            assert 0 <= row['promoted_rows'] <= 32
            assert row['old_hot_expired_after_preparation'] == [row['promoted_rows'] > 0]*2
        results[name] = dict(layer_checks=len(observations),contexts=4,teacher_steps_per_context=12,
                             prefix_bits_equal=True,capacity_fits=True,old_hot_owners_expire_after_promotion=True)
    # Historical trace grouping is descriptive. Verify intervals at each phase
    # peak and preserve shape/origin labels rather than claiming causal savings.
    resource_path = prior_dir / 'resource_shared.json'
    prior_proof = read(prior_dir / 'verified_parity_resource.json')
    assert sha(resource_path) == prior_proof['evidence_sha256']['resource_shared.json']
    resource = read(resource_path); groups = {}
    for phase,peak in resource['phase_peaks'].items():
        trace = next(t for t in resource['traces'] if t['step'] == peak['step'])
        sample = trace['samples'][peak['event']]
        assert {k:peak[k] for k in sample} == sample
        bins = collections.defaultdict(lambda:[0,0])
        for allocation in trace['allocations']:
            if allocation['birth_event'] <= peak['event'] and (allocation['end_event'] is None or allocation['end_event'] > peak['event']):
                key = (allocation['device'],allocation['origin'],allocation['dtype'],len(allocation['shape']))
                bins[key][0] += 1; bins[key][1] += allocation['bytes']
        assert sum(v[1] for v in bins.values()) == sample['known_live_kv_total_bytes']
        groups[phase] = [dict(device=k[0],origin=k[1],dtype=k[2],dimensions=k[3],storages=v[0],mib=v[1]/2**20)
                         for k,v in sorted(bins.items())]
    assert '206 passed' in (BASE / 'full_suite.txt').read_text()
    proof = dict(verified=True,geometries=results,total_layer_checks=3072,frozen_replay_rows=replayed,
                 historical_phase_peak_groups=groups,full_suite_passed=206,
                 advance_to_isolated_buffer_reuse_prototype=True,ordinary_candidate_ready=False,
                 measured_memory_or_speed_gain=False,complete_resource_equality=False,
                 evidence_sha256={n:sha(BASE/n) for n in ('plan.json','source.tar.gz','shared.json','default.json','jobs.json','full_suite.txt')},
                 historical_resource_sha256=sha(resource_path),verifier_sha256=sha(Path(__file__)),
                 limitations='Untimed read-only observer/reused teacher stream, one model/seed. Prefix identity/capacity are necessary conditions only: no in-place mutation, alias ordering, bit-safe compaction, lifetime stress, free generation, memory reduction or speed has been proved. Historical shape/origin groups describe known KV storage and exclude forward/metadata/allocator/driver scopes. No CLI/default/quality/physical STT/completion claim.')
    output.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(verified=True,total_layer_checks=3072,frozen_replay_rows=replayed,ordinary_candidate_ready=False),indent=2))


if __name__=='__main__': main()
