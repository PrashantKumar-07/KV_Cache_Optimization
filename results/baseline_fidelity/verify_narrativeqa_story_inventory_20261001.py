"""Independently confirm old-story QA identity and unused official test stories."""
import csv
import hashlib
import json
from collections import defaultdict,Counter
from pathlib import Path
import sys
sys.path[:0]=['experiments','src']
from audit_narrativeqa_story_ids import normalized_tokens
from audit_unused_documents import document_key
from provenance import read_jsonl
root=Path('results/baseline_fidelity')
raw=root/'narrativeqa_official_story_inventory_20261001.json'
output=root/'narrativeqa_official_story_inventory_verified_20261001.json'
assert not output.exists(),'Preserve prior evidence'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads(raw.read_text());metadata=Path('/data/nishant/tieredkv_external/narrativeqa_metadata_20261001')
m=json.loads((metadata/'manifest.json').read_text())
assert r['passed'] and r['runner_sha256']==sha('experiments/audit_narrativeqa_story_ids.py')
assert sha(metadata/'manifest.json')==r['metadata_manifest_sha256']
assert m['commit']==r['official_commit']
for name,entry in m['files'].items():assert sha(metadata/name)==entry['sha256']
assert sha(r['longbench_source_file'])==r['longbench_source_sha256']
with (metadata/'documents.csv').open(newline='') as stream:docs={d['document_id']:d for d in csv.DictReader(stream)}
with (metadata/'qaps.csv').open(newline='') as stream:qas=list(csv.DictReader(stream))
by_story=defaultdict(list)
for qa in qas:by_story[qa['document_id']].append(qa)
rows=read_jsonl(r['longbench_source_file']);mapping=r['normalized_context_to_official_story_ids']
assert len(rows)==r['longbench_rows']==r['matched_question_answer_rows']==200
assert len(mapping)==20 and r['unmatched_row_indices']==[]
for row in rows:
    ids=mapping[document_key(row)]
    assert len(ids)==1
    query=normalized_tokens(row['input']);answers={normalized_tokens(a) for a in row['answers']}
    assert any(query in {normalized_tokens(qa['question']),normalized_tokens(qa['question_tokenized'])}
               and answers & {normalized_tokens(qa[name]) for name in ('answer1','answer2','answer1_tokenized','answer2_tokenized')}
               for qa in by_story[ids[0]])
used={ids[0] for ids in mapping.values()}
assert len(used)==20
used_titles={normalized_tokens(docs[story]['wiki_title']) for story in used}
excluded=used|{story for story,doc in docs.items() if normalized_tokens(doc['wiki_title']) in used_titles}
assert excluded==set(r['excluded_official_story_ids'])
expected=[doc for doc in docs.values() if doc['set']=='test' and doc['document_id'] not in excluded]
assert expected==r['unused_test_stories'] and len(expected)==r['unused_test_story_count']==355
assert r['usable_length_inventory']=={kind:sum(doc['kind']==kind and 5000<=int(doc['story_word_count'])<=25000 for doc in expected) for kind in ('movie','gutenberg')}
report=dict(story_identity_exclusion_and_test_metadata_verified=True,raw_sha256=sha(raw),metadata_manifest_sha256=sha(metadata/'manifest.json'),
    official_repository=m['repository'],official_commit=m['commit'],old_contexts=20,old_questions=200,
    old_story_split_counts=dict(Counter(docs[story]['set'] for story in used)),unused_test_stories=355,
    metadata_length_eligible_stories=r['usable_length_inventory'],
    independent_story_download_or_evaluation_started=False,limitations=r['limitations'])
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
