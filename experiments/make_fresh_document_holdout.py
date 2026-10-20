"""Freeze 100 previously unused normalized documents before quality evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from provenance import atomic_json, read_jsonl
from audit_unused_documents import document_key, document_group_key, DOCUMENT_GROUP_SCHEMA

TASKS=('qasper','multifieldqa_en','hotpotqa','triviaqa','gov_report')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    inventory=json.loads(args.inventory.read_text())
    assert inventory['passed'] and inventory['tasks']['narrativeqa']['eligible_document_groups']==0
    if inventory.get('document_group_schema') != DOCUMENT_GROUP_SCHEMA:
        raise ValueError('Regenerate inventory with target-document identity before selection')
    rng=random.Random(20261003)
    used=set(inventory['excluded_document_group_keys'])
    chosen=set(); pending=[]
    manifest=dict(document_group_schema=DOCUMENT_GROUP_SCHEMA,model='../models/mistral-7b-instruct',seed=0,selection_seed=20261003,max_ctx=31500,
        arms=['gpu_proxy','gpu_query','cpu_query'],documents_per_task=20,
        inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),tasks={},
        excluded_task='narrativeqa: zero unused normalized documents in current source; independent task validation remains open',
        limitations=inventory['limitations'])
    for task in TASKS:
        meta=inventory['tasks'][task];source=Path(meta['source_file'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
        rows=read_jsonl(source)
        groups={key:indices for key,indices in meta['eligible_groups'].items() if key not in used|chosen}
        assert len(groups)>=20
        selected=sorted(rng.sample(sorted(groups),20))
        indices=sorted(rng.choice(groups[key]) for key in selected)
        data=[rows[i] for i in indices]
        keys=[document_group_key(row, task) for row in data]
        assert len(set(keys))==20 and not set(keys)&(used|chosen)
        assert set(keys)==set(selected)
        chosen.update(keys)
        dest=args.output.parent/'data'/f'{task}.jsonl'
        assert not dest.exists(),f'Preserve prior slice: {dest}'
        content=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in data)
        pending.append((dest,content))
        manifest['tasks'][task]=dict(source_file=str(source),source_sha256=meta['source_sha256'],source_row_count=len(rows),
            selected_zero_based_indices=indices,selected_sample_ids=[hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest() for row in data],
            selected_normalized_context_sha256=[document_key(row) for row in data],selected_document_group_sha256=keys,slice_file=str(dest),slice_sha256=hashlib.sha256(content.encode()).hexdigest())
    assert len(chosen)==100
    for dest,content in pending:
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x') as stream:stream.write(content)
    atomic_json(args.output,manifest)
    print(json.dumps({task:meta['selected_zero_based_indices'] for task,meta in manifest['tasks'].items()},indent=2))


if __name__=='__main__':
    main()
