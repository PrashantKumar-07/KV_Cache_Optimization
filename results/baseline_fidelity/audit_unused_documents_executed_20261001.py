"""Conservative exact/normalized-document inventory before a new quality holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from make_document_holdout import normalized
from provenance import atomic_json, read_jsonl


def sha(data):
    return hashlib.sha256(data).hexdigest()


def document_key(row):
    return sha(normalized(row['context']).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    base = Path('results/baseline_fidelity/real_query_holdout_20260929/manifest.json')
    previous = json.loads(base.read_text())
    evidence, used_ids = {}, set()
    for path in sorted(Path('results').rglob('*.json')):
        content = path.read_bytes()
        ids = set(re.findall(rb'"sample_id"\s*:\s*"([0-9a-f]{64})"', content))
        if ids:
            used_ids.update(x.decode() for x in ids)
            evidence[str(path)] = sha(content)
    manifests = {}
    excluded_contexts = set()
    for path in sorted(Path('results/baseline_fidelity').rglob('manifest.json')):
        content = path.read_bytes(); manifest = json.loads(content)
        manifests[str(path)] = sha(content)
        for task, meta in manifest.get('tasks', {}).items():
            if not isinstance(meta, dict) or 'source_file' not in meta:
                continue
            source = Path(meta['source_file'])
            rows = read_jsonl(source)
            assert sha(source.read_bytes()) == meta['source_sha256'], source
            prefix = meta.get('excluded_prior_prefix_count', 10 if task == 'gov_report' else 25)
            excluded = set(range(min(prefix,len(rows)))) | set(meta.get('selected_zero_based_indices', []))
            excluded_contexts.update(document_key(rows[i]) for i in excluded)
            used_ids.update(meta.get('selected_sample_ids', []))
    sources = {}
    folders = {Path(meta['source_file']).parent for meta in previous['tasks'].values()}
    for folder in folders:
        for task in previous['tasks']:
            for name in (f'{task}.jsonl', f'{task}_e.jsonl'):
                path = folder/name
                if not path.exists():
                    continue
                rows = read_jsonl(path)
                sources[str(path)] = sha(path.read_bytes())
                for row in rows:
                    sample = sha(json.dumps(row, sort_keys=True).encode())
                    if sample in used_ids:
                        excluded_contexts.add(document_key(row))
    tasks = {}
    for task, meta in previous['tasks'].items():
        source = Path(meta['source_file']); rows = read_jsonl(source)
        groups = {}
        for i,row in enumerate(rows):
            key = document_key(row)
            if key not in excluded_contexts:
                groups.setdefault(key,[]).append(i)
        tasks[task] = dict(source_file=str(source),source_sha256=sources[str(source)],source_row_count=len(rows),
            total_normalized_document_groups=len({document_key(row) for row in rows}),
            eligible_document_groups=len(groups),eligible_groups=groups)
    report = dict(passed=True,runner_sha256=sha(Path(__file__).read_bytes()),tasks=tasks,
        used_sample_id_count=len(used_ids),excluded_normalized_context_keys=sorted(excluded_contexts),
        evidence_with_sample_ids_sha256=evidence,prior_manifests_sha256=manifests,source_files_sha256=sources,
        selection_not_yet_frozen=True,
        limitations='Excludes recorded sample IDs, all prior manifest selections and conservative development prefixes, across standard/E source variants. Exact casefold/whitespace-normalized documents only; unrecorded manual evaluations, semantic near duplicates and model training overlap remain unverified. No independent evaluation has run.')
    atomic_json(args.output,report)
    print(json.dumps({task:{k:v for k,v in meta.items() if k not in ('eligible_groups','source_file','source_sha256')} for task,meta in tasks.items()},indent=2))


if __name__ == '__main__':
    main()
