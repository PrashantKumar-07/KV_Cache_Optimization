"""Freeze all original-source row partitions from a fresh recorded-use inventory."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from audit_unused_documents import document_group_key, DOCUMENT_GROUP_SCHEMA
from provenance import read_jsonl

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    out = Path(__file__).with_name('partitions.json')
    assert not out.exists(), 'Preserve frozen partitions'
    inventory_path = ROOT / 'results/audit_20261015_full_evaluation_partition_review.json'
    inventory = json.loads(inventory_path.read_text())
    assert inventory['passed'] and inventory['document_group_schema'] == DOCUMENT_GROUP_SCHEMA
    original_path = ROOT / 'results/remediation_v3/equal_vram_merged.json'
    original = json.loads(original_path.read_text())
    excluded = set(inventory['excluded_document_group_keys'])
    records = []
    summary = {}
    for task in original['config']['tasks']:
        meta = inventory['tasks'][task]
        source = ROOT / meta['source_file']
        assert sha(source) == meta['source_sha256'] == original['config']['dataset_sha256_by_task'][task]
        rows = read_jsonl(source)
        old_ids = [x['sample_id'] for x in original['results'][task]['full']['samples']]
        assert all([x['sample_id'] for x in arm['samples']] == old_ids
                   for arm in original['results'][task].values())
        task_records = []
        for index, row in enumerate(rows):
            sample_id = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            group = document_group_key(row, task)
            unconsumed = group not in excluded
            assert unconsumed == (index in meta['eligible_groups'].get(group, []))
            task_records.append(dict(task=task, source_zero_based_index=index,
                sample_id=sample_id, document_group_sha256=group,
                partition='unconsumed_as_of_inventory' if unconsumed else 'consumed_or_document_overlap',
                original135=sample_id in old_ids))
        assert [r['sample_id'] for r in task_records if r['original135']] == old_ids
        records.extend(task_records)
        summary[task] = dict(source_file=meta['source_file'], source_sha256=meta['source_sha256'],
            total_rows=len(rows), original135_rows=len(old_ids),
            consumed_or_document_overlap_rows=sum(r['partition']=='consumed_or_document_overlap' for r in task_records),
            unconsumed_as_of_inventory_rows=sum(r['partition']=='unconsumed_as_of_inventory' for r in task_records),
            unconsumed_document_groups=meta['eligible_document_groups'])
    # The late90 prediction screen must be included and may not contaminate the new stratum.
    quality_path = ROOT / 'results/audit_20261015_query_combination/quality.json'
    key = str(quality_path.relative_to(ROOT))
    assert inventory['evidence_with_sample_ids_sha256'][key] == sha(quality_path)
    import re
    late_ids = set(re.findall(r'"sample_id"\s*:\s*"([0-9a-f]{64})"', quality_path.read_text()))
    assert len(late_ids) == 30
    assert late_ids <= {r['sample_id'] for r in records if r['original135']}
    assert not late_ids & {r['sample_id'] for r in records if r['partition']=='unconsumed_as_of_inventory'}
    assert len(records) == 1474 and len({r['sample_id'] for r in records}) == 1474
    result = dict(schema='recorded_use_target_document_partition_v1', document_group_schema=DOCUMENT_GROUP_SCHEMA,
        inventory_file=str(inventory_path.relative_to(ROOT)), inventory_sha256=sha(inventory_path),
        original_file=str(original_path.relative_to(ROOT)), original_sha256=sha(original_path),
        generator_sha256=sha(__file__), tasks=summary, rows=records,
        late90_screen=dict(unique_sample_ids=30, all_in_original135=True, all_excluded_from_unconsumed=True,
            evidence_file=key, evidence_sha256=sha(quality_path)),
        limitation='Unconsumed means absent from recorded sample-ID/manifests and their normalized target-document overlap as of this inventory. It does not prove absence of unrecorded/manual exposure, semantic overlap, or model-training exposure. NarrativeQA has no unconsumed group. Full1474 union is descriptive; preserve unconsumed rows until final source/config/analysis lock and do not tune after unblinding.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
