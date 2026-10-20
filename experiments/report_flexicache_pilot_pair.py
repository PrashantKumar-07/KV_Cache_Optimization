"""Validate two adapted FlexiCache feasibility runs; keep resource scopes separate."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from provenance import atomic_json


TOKEN_HASH = '54e41e6b9f3ef83377ff4985916145521fc1eb1ebe33a985cb503d275c10a890'
COMMIT = 'ab0f495b10fc6509c1637af35779b29121fb7d3f'


def summarize(first, repeat):
    for run in (first, repeat):
        if run.get('passed') is not True:
            raise ValueError('Both fresh-process model runs must pass')
        if run['upstream_commit'] != COMMIT or run['upstream_diff']:
            raise ValueError('Expected unchanged released source')
        if run['input_token_sha256'] != TOKEN_HASH or run['input_tokens'] != 1440:
            raise ValueError('Expected the frozen identical input tokens')
        if run['generated_tokens'] != len(run['generated_token_ids']):
            raise ValueError('Generated token count disagrees with IDs')
        if run['runtime_resolver_adapter']['query_return_code'] != 0:
            raise ValueError('Runtime adapter was not validated')
        for pool in ('gpu_kv_cache_pool', 'cpu_kv_cache_pool'):
            info = run['allocated_pools'][pool]
            if info['dtype'] != 'torch.bfloat16' or math.prod(info['shape']) * 2 != info['allocated_payload_bytes']:
                raise ValueError('Invalid bf16 pool payload')
        for call in run['prefill_launch_adapter']['calls']:
            if call['num_stages'] != 1 or call['compiled_shared_bytes'] > 101376:
                raise ValueError('Invalid L40S launch adaptation')
        obs = run['resource_observations']
        for counts in obs['transfers'].values():
            if counts['payload_bytes'] != counts['paired_kv_pages'] * 8192:
                raise ValueError('Transfer payload disagrees with bf16 16x128 paired pages')
    exact_fields = ('upstream_commit', 'upstream_diff', 'runner_sha256', 'model_config_sha256',
        'data_sha256', 'pool_config_sha256', 'sample_id', 'prompt_sha256',
        'input_token_sha256', 'input_tokens', 'engine_environment', 'engine_settings',
        'vllm_version', 'runtime', 'runtime_resolver_adapter', 'prefill_launch_adapter',
        'transfer_extensions', 'allocated_pools', 'prediction', 'generated_token_ids',
        'generated_tokens', 'score')
    for field in exact_fields:
        if first[field] != repeat[field]:
            raise ValueError(f'Fresh-process reproducibility mismatch: {field}')
    pools, obs = first['allocated_pools'], first['resource_observations']
    snapshots = obs['scheduler_snapshots']
    repeated_obs = repeat['resource_observations']
    repeated_snapshots = repeated_obs['scheduler_snapshots']
    if obs['transfers'] != repeated_obs['transfers']:
        raise ValueError('Submitted transfer payload differs')
    if [row for row in snapshots if row['scheduled_tokens']] != [row for row in repeated_snapshots if row['scheduled_tokens']]:
        raise ValueError('Allocation trace for scheduled tokens differs')
    limits = {'gpu_pages_in_use': pools['gpu_kv_cache_pool']['shape'][1],
              'cpu_pages_in_use': pools['cpu_kv_cache_pool']['shape'][1],
              'minmax_pages_in_use': math.prod(pools['minmax_key_cache']['shape'][:2])}
    for row in snapshots + repeated_snapshots:
        if any(not 0 <= row[key] <= limit for key, limit in limits.items()):
            raise ValueError('Scheduler allocation exceeds its tensor pool')
    return {'passed': True, 'scope': 'one reproducible adapted feasibility request; resources unmatched',
        'baseline': 'FlexiCache L40S port with recorded runtime and prefill-launch adapters',
        'identical_input_output_pools_and_payload': True,
        'full_scheduler_trace_equal': obs == repeated_obs,
        'nonempty_scheduled_allocation_trace_equal': True,
        'input_tokens': first['input_tokens'], 'input_token_sha256': TOKEN_HASH,
        'generated_tokens': first['generated_tokens'], 'score': first['score'],
        'resource_matched': False,
        'reserved_tensor_pool_bytes': {name: info['allocated_payload_bytes'] for name, info in pools.items()},
        'submitted_kv_transfers': obs['transfers'],
        'scheduler_allocation_snapshots': {'counts': [len(snapshots), len(repeated_snapshots)],
            'idle_poll_counts': [sum(row['scheduled_tokens'] == 0 for row in rows) for rows in (snapshots, repeated_snapshots)],
            'first': snapshots[0], 'last': snapshots[-1],
            'max_sampled_pages': {key: max(row[key] for row in snapshots) for key in limits},
            'scope': 'allocated block snapshots, not filled useful KV occupancy or continuous peaks'},
        'observed_whole_request_seconds': [run['whole_request_s'] for run in (first, repeat)],
        'generation_peak_gpu_allocated_bytes': [run['gpu_peak_allocated_bytes'] for run in (first, repeat)],
        'process_peak_rss_bytes': [run['process_peak_rss_bytes'] for run in (first, repeat)],
        'limitations': 'Single short request; reserved pools, allocations and submitted payload are distinct. Launch adaptation is reference-checked, not bitwise compared with unsupported original launch. Timing includes observers/compilation; no quality, speed, energy, STT endurance or matched-resource claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--repeat', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    result = summarize(json.loads(args.first.read_text()), json.loads(args.repeat.read_text()))
    result['raw_inputs'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.first, args.repeat)}
    result['reporter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(args.output, result)
    print(json.dumps({'passed': True, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
