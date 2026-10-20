"""Verify frozen 100-document identity, exclusion and deterministic sampling."""
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path[:0]=['experiments','src']
from provenance import read_jsonl
from audit_unused_documents import document_key
root=Path('results/baseline_fidelity/realq_cpu_fresh_documents_20261001')
output=root/'verified_manifest.json'
assert not output.exists(),'Preserve prior evidence'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=root/'manifest.json';m=json.loads(p.read_text())
inventory=Path('results/baseline_fidelity/unused_document_inventory_20261001.json');i=json.loads(inventory.read_text())
assert m['inventory_sha256']==sha(inventory)
assert m['runner_sha256']==sha('experiments/make_fresh_document_holdout.py')
assert i['runner_sha256']==sha('experiments/audit_unused_documents.py')
for category in ('evidence_with_sample_ids_sha256','prior_manifests_sha256','source_files_sha256'):
    for path,expected in i[category].items():assert sha(path)==expected,path
assert i['tasks']['narrativeqa']['eligible_document_groups']==0
assert list(m['tasks'])==['qasper','multifieldqa_en','hotpotqa','triviaqa','gov_report']
assert m['selection_seed']==20261003 and m['documents_per_task']==20
rng=random.Random(m['selection_seed']);used=set(i['excluded_normalized_context_keys']);chosen=set()
for task,entry in m['tasks'].items():
    source=Path(entry['source_file']);rows=read_jsonl(source);data=read_jsonl(entry['slice_file'])
    assert sha(source)==entry['source_sha256'] and sha(entry['slice_file'])==entry['slice_sha256']
    assert len(rows)==entry['source_row_count'] and len(data)==20
    assert data==[rows[n] for n in entry['selected_zero_based_indices']]
    keys=[document_key(row) for row in data]
    assert keys==entry['selected_normalized_context_sha256'] and len(set(keys))==20
    assert not set(keys)&(used|chosen)
    assert [hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest() for row in data]==entry['selected_sample_ids']
    assert all(key in i['tasks'][task]['eligible_groups'] for key in keys)
    groups={key:indices for key,indices in i['tasks'][task]['eligible_groups'].items() if key not in used|chosen}
    expected=sorted(rng.sample(sorted(groups),20))
    assert sorted(keys)==expected
    assert entry['selected_zero_based_indices']==sorted(rng.choice(groups[key]) for key in expected)
    chosen.update(keys)
assert len(chosen)==100
report=dict(frozen_source_slice_identity_exclusion_and_sampling_verified=True,manifest_sha256=sha(p),inventory_sha256=sha(inventory),
    documents=100,tasks=5,per_task=20,semantic_near_duplicate_or_training_overlap_verified=False,
    narrativeqa_unused_groups=0,evaluation_started=False,
    limitations='Disjoint from recorded IDs/manifests/development prefixes under exact casefold/whitespace context identity. Unrecorded manual evaluations, semantic duplicates and training contamination unverified. NarrativeQA requires additional documents before independent validation.')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
