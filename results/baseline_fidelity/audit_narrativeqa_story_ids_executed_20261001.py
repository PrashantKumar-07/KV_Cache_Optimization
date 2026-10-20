"""Identify already-used LongBench stories in frozen official NarrativeQA metadata."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from provenance import atomic_json, read_jsonl
from audit_unused_documents import document_key


def normalized_tokens(value):
    return ' '.join(re.findall(r'\w+',value.casefold()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    metadata_manifest=args.metadata/'manifest.json'
    m=json.loads(metadata_manifest.read_text())
    for name,meta in m['files'].items():
        assert hashlib.sha256((args.metadata/name).read_bytes()).hexdigest()==meta['sha256']
    with (args.metadata/'documents.csv').open(newline='') as f:documents=list(csv.DictReader(f))
    with (args.metadata/'qaps.csv').open(newline='') as f:questions=list(csv.DictReader(f))
    by_id={doc['document_id']:doc for doc in documents}
    assert len(documents)==len(by_id)==1572
    lookup=defaultdict(set)
    for row in questions:
        assert row['document_id'] in by_id
        for qfield in ('question','question_tokenized'):
            for afield in ('answer1','answer2','answer1_tokenized','answer2_tokenized'):
                lookup[(normalized_tokens(row[qfield]),normalized_tokens(row[afield]))].add(row['document_id'])
    previous=json.loads(Path('results/baseline_fidelity/real_query_holdout_20260929/manifest.json').read_text())['tasks']['narrativeqa']
    source=Path(previous['source_file'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==previous['source_sha256']
    rows=read_jsonl(source)
    groups=defaultdict(set);mapped_rows=0;unmatched=[]
    for index,row in enumerate(rows):
        key=document_key(row)
        matches=set()
        for answer in row['answers']:
            matches.update(lookup[(normalized_tokens(row['input']),normalized_tokens(answer))])
        groups[key].update(matches)
        if matches:mapped_rows+=1
        else:unmatched.append(index)
    assert len(groups)==20
    assert all(len(ids)==1 for ids in groups.values()), 'Story identity ambiguous/missing; do not claim unused documents'
    excluded=set().union(*groups.values())
    assert len(excluded)==20
    used_titles={normalized_tokens(by_id[doc]['wiki_title']) for doc in excluded}
    excluded.update(doc['document_id'] for doc in documents if normalized_tokens(doc['wiki_title']) in used_titles)
    test=[doc for doc in documents if doc['set']=='test' and doc['document_id'] not in excluded]
    report=dict(passed=True,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_repository=m['repository'],official_commit=m['commit'],metadata_manifest_sha256=hashlib.sha256(metadata_manifest.read_bytes()).hexdigest(),
        longbench_source_file=str(source),longbench_source_sha256=previous['source_sha256'],longbench_rows=len(rows),matched_question_answer_rows=mapped_rows,
        unmatched_row_indices=unmatched,normalized_context_to_official_story_ids={key:sorted(ids) for key,ids in groups.items()},
        excluded_official_story_ids=sorted(excluded),unused_test_story_count=len(test),unused_test_stories=test,
        usable_length_inventory={kind:sum(doc['kind']==kind and 5000<=int(doc['story_word_count'])<=25000 for doc in test) for kind in ('movie','gutenberg')},
        story_download_or_evaluation_started=False,
        limitations='Question+answer identity maps old context groups to official story IDs; excludes those IDs and normalized Wikipedia titles. Raw story downloads, current availability, cleanup/content integrity, semantic near duplicates, actual token lengths and model training overlap remain unverified. This is metadata availability, not a frozen evaluation or independent quality evidence.')
    atomic_json(args.output,report)
    print(json.dumps({key:report[key] for key in ('official_commit','longbench_rows','matched_question_answer_rows','unused_test_story_count','usable_length_inventory')},indent=2))


if __name__=='__main__':
    main()
