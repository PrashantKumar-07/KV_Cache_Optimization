"""Freeze a new 20-story NarrativeQA test preflight before downloading content."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from provenance import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    inventory=json.loads(args.inventory.read_text());assert inventory['passed']
    manifest=json.loads((args.metadata/'manifest.json').read_text())
    assert hashlib.sha256((args.metadata/'manifest.json').read_bytes()).hexdigest()==inventory['metadata_manifest_sha256']
    with (args.metadata/'qaps.csv').open(newline='') as stream:qas=list(csv.DictReader(stream))
    assert hashlib.sha256((args.metadata/'qaps.csv').read_bytes()).hexdigest()==manifest['files']['qaps.csv']['sha256']
    by_story=defaultdict(list)
    for qa in qas:by_story[qa['document_id']].append(qa)
    rng=random.Random(20261004);selected=[]
    for kind in ('gutenberg','movie'):
        eligible=sorted([doc for doc in inventory['unused_test_stories']
            if doc['kind']==kind and 5000<=int(doc['story_word_count'])<=25000],key=lambda doc:doc['document_id'])
        chosen=sorted(rng.sample(eligible,10),key=lambda doc:doc['document_id'])
        for doc in chosen:
            assert doc['set']=='test' and doc['document_id'] not in inventory['excluded_official_story_ids']
            questions=sorted(by_story[doc['document_id']],key=lambda row:(row['question'],row['answer1'],row['answer2']))
            qa=rng.choice(questions);assert qa['set']=='test'
            selected.append(dict(story=doc,qa=qa))
    assert len(selected)==len({row['story']['document_id'] for row in selected})==20
    report=dict(passed=True,selection_seed=20261004,inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        metadata_manifest_sha256=inventory['metadata_manifest_sha256'],official_commit=manifest['commit'],
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selected=selected,
        selection_rule='10 Gutenberg + 10 movie test stories, metadata5K–25Kwords, sorted IDs before seeded sampling; one seeded QA per story; no replacement for download failures',
        download_started=False,model_evaluation_started=False)
    atomic_json(args.output,report)
    print(json.dumps({'selected_stories':20,'selection_seed':report['selection_seed'],'manifest_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':
    main()
