"""Verify 20 story/QA selections were deterministic and disjoint before fetch."""
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
root=Path('results/baseline_fidelity/narrativeqa_test_stories_20261001')
output=root/'verified_selection.json';assert not output.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((root/'manifest.json').read_text())
i_path=Path('results/baseline_fidelity/narrativeqa_official_story_inventory_20261001.json');i=json.loads(i_path.read_text())
metadata=Path('/data/nishant/tieredkv_external/narrativeqa_metadata_20261001')
meta=json.loads((metadata/'manifest.json').read_text())
assert m['inventory_sha256']==sha(i_path) and m['metadata_manifest_sha256']==sha(metadata/'manifest.json')
assert m['official_commit']==meta['commit']==i['official_commit']
assert m['runner_sha256']==sha(root/'prepare_narrativeqa_test_stories_executed.py')
assert sha(metadata/'qaps.csv')==meta['files']['qaps.csv']['sha256']
with (metadata/'qaps.csv').open(newline='') as stream:qas=list(csv.DictReader(stream))
by_story=defaultdict(list)
for qa in qas:by_story[qa['document_id']].append(qa)
rng=random.Random(20261004);expected=[]
for kind in ('gutenberg','movie'):
    eligible=sorted([doc for doc in i['unused_test_stories'] if doc['kind']==kind and 5000<=int(doc['story_word_count'])<=25000],key=lambda d:d['document_id'])
    chosen=sorted(rng.sample(eligible,10),key=lambda d:d['document_id'])
    for doc in chosen:
        qa=rng.choice(sorted(by_story[doc['document_id']],key=lambda q:(q['question'],q['answer1'],q['answer2'])))
        expected.append(dict(story=doc,qa=qa))
assert expected==m['selected'] and m['selection_seed']==20261004
ids={row['story']['document_id'] for row in m['selected']}
assert len(ids)==20 and not ids&set(i['excluded_official_story_ids'])
assert all(row['story']['set']==row['qa']['set']=='test' for row in m['selected'])
report=dict(story_qa_source_sampling_and_exclusion_verified=True,manifest_sha256=sha(root/'manifest.json'),
    stories=20,books=10,movies=10,source_commit=m['official_commit'],
    no_model_quality_evaluation=True,limitations='Selection identity only. Downloaded content/encoding/marker/token checks still need terminal preflight verification; no semantic/training overlap proof.')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
