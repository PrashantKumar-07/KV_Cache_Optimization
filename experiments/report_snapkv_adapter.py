"""Validate adapter pilot provenance and summarize scores and stored KV payload."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from provenance import atomic_json, runtime_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input')
    args = p.parse_args()
    source = Path(args.input)
    doc = json.loads(source.read_text())
    config = doc['config']
    assert config['runtime'] == runtime_identity(), 'Runtime/source changed'
    for filename, expected in config['source_files'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    import longbench_eval as bench
    for task, expected in config['dataset_sha256'].items():
        assert hashlib.sha256(bench.task_data_path(config['data_dir'], task).read_bytes()).hexdigest() == expected
    assert set(doc['results']) == set(config['tasks']), 'Incomplete tasks'
    model_config = json.loads((Path(config['model']) / 'config.json').read_text())
    heads = model_config['num_attention_heads']
    head_dim = model_config.get('head_dim', model_config['hidden_size'] // heads)
    # Driver uses bf16 and upstream expanded query-head storage for both K and V.
    bytes_per_token = 2 * 2 * model_config['num_hidden_layers'] * heads * head_dim
    summary = {}
    for task, entries in doc['results'].items():
        entry = entries['snapkv_upstream_selector']
        rows = entry['samples']
        assert len(rows) == entry['n'] == config['max_samples']
        assert len({r['sample_id'] for r in rows}) == len(rows)
        # JSON strings may contain Unicode line separators; only physical
        # JSONL newlines delimit records, as in the evaluation loader.
        with bench.task_data_path(config['data_dir'], task).open(encoding='utf-8') as stream:
            data = [json.loads(line) for line in stream if line.strip()]
        expected_ids = [hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                        for item in data[:config['max_samples']]]
        assert [r['sample_id'] for r in rows] == expected_ids
        metric = bench.DATASET_TO_METRIC[task]
        for row, item in zip(rows, data):
            prediction = row['prediction']
            if task in ('trec', 'triviaqa', 'samsum', 'lsht'):
                prediction = prediction.lstrip('\n').split('\n')[0]
            recomputed = 100 * max(metric(prediction, answer,
                all_classes=item.get('all_classes', [])) for answer in item['answers'])
            assert abs(recomputed - row['score']) < 1e-8, 'Saved prediction score mismatch'
        assert entry['sample_scores'] == [round(row['score'], 2) for row in rows]
        assert abs(np.mean([r['score'] for r in rows]) - entry['score']) < .011
        assert all(r['final_kv_payload_bytes'] > 0 for r in rows)
        for row in rows:
            final_length = min(row['prompt_tokens'], config['capacity']) + row['generated_tokens'] - 1
            assert row['final_kv_payload_bytes'] == final_length * bytes_per_token, 'KV byte accounting mismatch'
        summary[task] = {'score':entry['score'], 'n':len(rows),
            'mean_final_kv_payload_bytes':float(np.mean([r['final_kv_payload_bytes'] for r in rows]))}
    report = {'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'tasks':summary, 'expanded_kv_bytes_per_token':bytes_per_token,
              'model_config_sha256':hashlib.sha256((Path(config['model']) / 'config.json').read_bytes()).hexdigest(),
              'scope':'Adapter pilot; expanded-head KV and growing decode residency. '
                      'Not equal-byte or fixed-attended-width comparison; no speed claim.'}
    atomic_json(source.with_name(source.stem + '_report.json'), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
