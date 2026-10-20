"""Audit the four-example configured-byte/attention-width SnapKV pilot."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / 'results/baseline_fidelity/matched_pilot_snap_eager.json'
TIERED = ROOT / 'results/baseline_fidelity/matched_pilot_tiered_bytes.json'
OUT = ROOT / 'results/baseline_fidelity/matched_pilot_report.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    snap, tiered = json.loads(SNAP.read_text()), json.loads(TIERED.read_text())
    sc, tc = snap['config'], tiered['config']
    assert sc['runtime'] == tc['runtime']
    assert sc['dataset_sha256'] == tc['dataset_sha256']
    assert sc['tasks'] == tc['tasks'] == ['qasper', 'gov_report']
    assert sc['max_samples'] == tc['max_samples'] == 2
    assert sc['attn_implementation'] == 'eager' and tc['strategy'] == 'query'
    assert tc['tiered_vram_budget'] + tc['tiered_stt_budget'] == 4 * sc['capacity'] == 4096
    assert tc['tiered_vram_budget'] + tc['stt_expose_quota'] == sc['capacity'] == 1024
    model = json.loads((ROOT.parent / 'models/mistral-7b-instruct/config.json').read_text())
    row = 4 * model['num_hidden_layers'] * model['num_attention_heads'] * (
        model.get('head_dim', model['hidden_size'] // model['num_attention_heads']))
    assert row == 524288
    tasks = {}
    for task in sc['tasks']:
        s = snap['results'][task]['snapkv_upstream_selector']['samples']
        t = tiered['results'][task]['tieredkv']['samples']
        assert len(s) == len(t) == 2
        assert [x['sample_id'] for x in s] == [x['sample_id'] for x in t]
        assert [x['prompt_tokens'] for x in s] == [x['prompt_tokens'] for x in t]
        assert all(x['final_kv_payload_bytes'] ==
                   (min(x['prompt_tokens'], 1024) + x['generated_tokens'] - 1) * row
                   for x in s)
        assert all(0 < x['final_resident_kv_payload_bytes'] <= 4096 * row // 4
                   for x in t)
        differences = [a['score'] - b['score'] for a, b in zip(t, s)]
        tasks[task] = {'n': 2, 'snapkv_score': snap['results'][task]['snapkv_upstream_selector']['score'],
                       'tiered_score': tiered['results'][task]['tieredkv']['score'],
                       'paired_mean_difference': sum(differences) / 2,
                       'paired_differences': differences,
                       'snapkv_final_kv_bytes': [x['final_kv_payload_bytes'] for x in s],
                       'tiered_final_resident_kv_bytes':
                       [x['final_resident_kv_payload_bytes'] for x in t],
                       'tiered_to_snapkv_final_byte_ratio':
                       [a['final_resident_kv_payload_bytes'] / b['final_kv_payload_bytes']
                        for a, b in zip(t, s)]}
    report = {
        'source_sha256': {str(SNAP.relative_to(ROOT)): sha(SNAP),
                          str(TIERED.relative_to(ROOT)): sha(TIERED)},
        'script_sha256': sha(Path(__file__)),
        'shared_runtime': sc['runtime'], 'tasks': tasks,
        'task_balanced_mean_difference': sum(x['paired_mean_difference'] for x in tasks.values()) / 2,
        'configured_prompt_payload_ceiling_bytes_each': sc['capacity'] * row,
        'attended_rows_ceiling_each': sc['capacity'],
        'decision': 'Do not advance this configuration from the four-example screen.',
        'limitations': ['Configured prompt ceilings match, but actual final resident KV bytes differ.',
                        'SnapKV appends decode rows; TieredKV keeps a fixed resident budget.',
                        'Query-aware TieredKV scans cold keys; SnapKV uses a prompt selector.',
                        'Four examples cannot establish quality non-inferiority or superiority; no latency claim.']}
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'tasks': tasks, 'task_balanced_mean_difference':
                      report['task_balanced_mean_difference']}, indent=2))


if __name__ == '__main__':
    main()
