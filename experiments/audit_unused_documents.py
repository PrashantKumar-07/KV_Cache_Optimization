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


DOCUMENT_GROUP_SCHEMA = "target_document_v2"


def document_group_key(row, task):
    """Exclude the target document, not variable few-shot demonstrations.

    Historical document_key remains unchanged for frozen evidence readers.
    """
    target = row['context']
    if task == 'triviaqa':
        target = row['input'].rsplit('\nQuestion:', 1)[0]
    elif task in ('samsum', 'trec'):
        target = row['input']
    return sha((task + '\0' + normalized(target)).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, help='Relocated source directory; exact recorded hash is still required')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    base = Path('results/baseline_fidelity/real_query_holdout_20260929/manifest.json')
    previous = json.loads(base.read_text())
    def source_path(meta):
        recorded = Path(meta['source_file'])
        return recorded if recorded.exists() else args.data_dir / recorded.name if args.data_dir else recorded
    evidence, used_ids = {}, set()
    for path in sorted(Path('results').rglob('*.json')):
        content = path.read_bytes()
        ids = set(re.findall(rb'"sample_id"\s*:\s*"([0-9a-f]{64})"', content))
        if ids:
            used_ids.update(x.decode() for x in ids)
            evidence[str(path)] = sha(content)
    manifests = {}
    excluded_contexts = set()
    excluded_groups = set()
    for path in sorted(Path('results/baseline_fidelity').rglob('manifest.json')):
        content = path.read_bytes(); manifest = json.loads(content)
        manifests[str(path)] = sha(content)
        for task, meta in manifest.get('tasks', {}).items():
            if not isinstance(meta, dict) or 'source_file' not in meta:
                continue
            source = source_path(meta)
            rows = read_jsonl(source)
            assert sha(source.read_bytes()) == meta['source_sha256'], source
            prefix = meta.get('excluded_prior_prefix_count', 10 if task == 'gov_report' else 25)
            excluded = set(range(min(prefix,len(rows)))) | set(meta.get('selected_zero_based_indices', []))
            excluded_contexts.update(document_key(rows[i]) for i in excluded)
            excluded_groups.update(document_group_key(rows[i], task) for i in excluded)
            used_ids.update(meta.get('selected_sample_ids', []))
    sources = {}
    folders = {source_path(meta).parent for meta in previous['tasks'].values()}
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
                        excluded_groups.add(document_group_key(row, task))
    tasks = {}
    for task, meta in previous['tasks'].items():
        source = source_path(meta); rows = read_jsonl(source)
        groups = {}
        for i,row in enumerate(rows):
            key = document_group_key(row, task)
            if key not in excluded_groups:
                groups.setdefault(key,[]).append(i)
        tasks[task] = dict(source_file=str(source),source_sha256=sources[str(source)],source_row_count=len(rows),
            total_normalized_document_groups=len({document_group_key(row, task) for row in rows}),
            eligible_document_groups=len(groups),eligible_groups=groups)
    report = dict(passed=True,document_group_schema=DOCUMENT_GROUP_SCHEMA,
        excluded_document_group_keys=sorted(excluded_groups),runner_sha256=sha(Path(__file__).read_bytes()),tasks=tasks,
        used_sample_id_count=len(used_ids),excluded_normalized_context_keys=sorted(excluded_contexts),
        evidence_with_sample_ids_sha256=evidence,prior_manifests_sha256=manifests,source_files_sha256=sources,
        selection_not_yet_frozen=True,
        limitations='Excludes recorded sample IDs, all prior manifest selections and conservative development prefixes, across standard/E source variants. Task-aware casefold/whitespace-normalized target documents only; TriviaQA demonstrations are excluded from identity; unrecorded manual evaluations, semantic near duplicates and model training overlap remain unverified. No independent evaluation has run.')
    atomic_json(args.output,report)
    print(json.dumps({task:{k:v for k,v in meta.items() if k not in ('eligible_groups','source_file','source_sha256')} for task,meta in tasks.items()},indent=2))


if __name__ == '__main__':
    main()
