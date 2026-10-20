"""Reserve novel benchmark documents without loading any prediction scores."""
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'hf_cache/datasets/downloads/extracted/5def9c7eaf83951e1cee75902d8611ef764ad88dd289e8f0bf97ff658a3eb83a/data'


def sha(content):
    return hashlib.sha256(content).hexdigest()


def context_key(row):
    return sha(' '.join(row['context'].casefold().split()).encode())


def main():
    destination = OUT / 'fresh_document_reserve.json'
    if destination.exists():
        raise FileExistsError(destination)
    used_ids = set()
    evidence = {}
    # Includes selected_sample_ids lists in reserved manifests. Other SHA256
    # values are harmless: only exact dataset-row hashes are looked up later.
    pattern = re.compile(rb'"([0-9a-f]{64})"')
    for path in sorted((ROOT / 'results').rglob('*')):
        if path.suffix not in ('.json', '.jsonl') or OUT in path.parents:
            continue
        raw = path.read_bytes()
        ids = pattern.findall(raw)
        if ids:
            used_ids.update(value.decode() for value in ids)
            evidence[str(path.relative_to(ROOT))] = sha(raw)
    # Every six-task current-final source is consumed even if a checkpoint
    # stream is relocated. Exclude exact normalized contexts across tasks.
    consumed_contexts = set()
    current = json.loads((ROOT / 'results/full_evaluation_20261015/partitions.json').read_text())
    for meta in current['tasks'].values():
        for line in (ROOT / meta['source_file']).read_text().split('\n'):
            if not line.strip():
                continue
            row = json.loads(line)
            consumed_contexts.add(context_key(row))
            used_ids.add(sha(json.dumps(row, sort_keys=True).encode()))
    sources = {}
    for path in sorted(DATA.glob('*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().split('\n') if line.strip()]
        sources[path.name] = rows
        for row in rows:
            if sha(json.dumps(row, sort_keys=True).encode()) in used_ids:
                consumed_contexts.add(context_key(row))
    tasks = {}
    for task in ('2wikimqa', 'musique', 'qmsum', 'multi_news'):
        path = DATA / f'{task}.jsonl'
        rows = sources[path.name]
        groups = {}
        for index, row in enumerate(rows):
            sample = sha(json.dumps(row, sort_keys=True).encode())
            context = context_key(row)
            if sample not in used_ids and context not in consumed_contexts:
                groups.setdefault(context, []).append(dict(source_index=index, sample_id=sample))
        # Entire eligible population is reserved. No development sub-split is
        # opened: tuning remains restricted to the original 135 records.
        tasks[task] = dict(source_file=str(path), source_sha256=sha(path.read_bytes()),
            population_rows=len(rows), reserved_document_groups=len(groups),
            reserved_rows=sum(map(len, groups.values())), reserved_groups=groups)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Prospective independent validation reserve; do not use for candidate selection or tuning',
        development='Only original135 historical IDs, not remaining1339 consumed current rows',
        status='Candidate-independent data reservation, not an approved evaluation launch protocol',
        grouping='Cross-task casefold/whitespace-normalized whole-context equality',
        tasks=tasks, historical_sample_ids=len(used_ids),
        evidence_files_sha256=evidence, runner_sha256=sha(Path(__file__).read_bytes()),
        limitations=['Unrecorded manual evaluations cannot be excluded.',
            'Semantic near duplicates and partially overlapping Wikipedia source passages are not excluded.',
            'Task/domain shift prevents direct confirmation of original six-task aggregate superiority.',
            'Recorded reserved IDs in previous evidence are conservatively excluded even if never scored.'])
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({task:{key:value for key,value in meta.items() if key!='reserved_groups'} for task,meta in tasks.items()}, indent=2))


if __name__ == '__main__':
    main()
