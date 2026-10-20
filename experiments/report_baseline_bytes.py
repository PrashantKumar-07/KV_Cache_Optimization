"""Report the stored-KV byte mismatch in the completed SnapKV comparison."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAP = [ROOT / 'results/snapkv_adapter_full/short.json',
        ROOT / 'results/snapkv_adapter_full/gov.json']
TIERED = ROOT / 'results/remediation_v3/equal_vram_merged.json'
OUT = ROOT / 'results/baseline_fidelity/snapkv_byte_comparison.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    model = json.loads((ROOT.parent / 'models/mistral-7b-instruct/config.json').read_text())
    layers = model['num_hidden_layers']
    head_dim = model.get('head_dim', model['hidden_size'] // model['num_attention_heads'])
    qheads = model['num_attention_heads']
    kvheads = model['num_key_value_heads']
    snap_row = 2 * 2 * layers * qheads * head_dim
    tier_row = 2 * 2 * layers * kvheads * head_dim
    tier_doc = json.loads(TIERED.read_text())
    assert tier_doc['config']['tiered_kv_dtype'] == 'bfloat16'
    slots = tier_doc['config']['tiered_vram_budget'] + tier_doc['config']['tiered_stt_budget']
    assert slots == 3072 and qheads == 4 * kvheads
    tasks = {}
    source_sha = {}
    for path in SNAP:
        doc = json.loads(path.read_text())
        assert doc['config']['capacity'] == 1024
        source_sha[str(path.relative_to(ROOT))] = sha(path)
        for task, methods in doc['results'].items():
            rows = methods['snapkv_upstream_selector']['samples']
            lengths = [min(r['prompt_tokens'], 1024) + r['generated_tokens'] - 1 for r in rows]
            assert all(r['final_kv_payload_bytes'] == n * snap_row
                       for r, n in zip(rows, lengths))
            bytes_ = [r['final_kv_payload_bytes'] for r in rows]
            tasks[task] = {'n': len(rows), 'mean_final_snapkv_kv_bytes': sum(bytes_) / len(bytes_),
                           'min_final_snapkv_kv_bytes': min(bytes_),
                           'max_final_snapkv_kv_bytes': max(bytes_),
                           'fraction_exceeding_tiered_resident_byte_ceiling':
                           sum(b > slots * tier_row for b in bytes_) / len(bytes_)}
    source_sha[str(TIERED.relative_to(ROOT))] = sha(TIERED)
    report = {'input_sha256': source_sha, 'report_script_sha256': sha(Path(__file__)),
              'snapkv_bytes_per_token': snap_row,
              'tiered_bytes_per_token': tier_row,
              'tiered_resident_row_ceiling': slots,
              'tiered_resident_payload_byte_ceiling': slots * tier_row,
              'tasks': tasks,
              'scope': 'KV payload only; SnapKV final measured bytes versus TieredKV configured '
                       'resident byte ceiling. Excludes model weights, temporary tensors, metadata, '
                       'full prefill peak, and differing attended width. Not an equal-byte result.'}
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
