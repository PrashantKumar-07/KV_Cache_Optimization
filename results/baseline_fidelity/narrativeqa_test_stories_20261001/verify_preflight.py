"""Check all frozen downloads and independently recount usable coverage."""
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=['experiments','src']
from download_narrativeqa_test_stories import visible_text,marker_positions,LIMIT
root=Path('results/baseline_fidelity/narrativeqa_test_stories_20261001')
output=root/'verified_preflight.json';assert not output.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((root/'manifest.json').read_text());r=json.loads((root/'preflight.json').read_text())
assert r['preflight_completed'] and len(r['rows'])==20
assert r['manifest_sha256']==sha(root/'manifest.json')
assert r['runner_sha256']==sha(root/'download_narrativeqa_test_stories_executed.py')
assert r['runner_sha256']==sha('experiments/download_narrativeqa_test_stories.py')
assert r['observation_dependency_sha256']==sha('experiments/audit_narrativeqa_story_ids.py')
assert [row['story_id'] for row in r['rows']]==[selected['story']['document_id'] for selected in m['selected']]
usable=[];failures=[]
for row,selected in zip(r['rows'],m['selected']):
    story=selected['story']
    assert row['kind']==story['kind'] and row['attempts'][0]['requested_url']==story['story_url']
    assert 1<=len(row['attempts'])<=2
    if 'raw_file' in row:
        body=Path(row['raw_file']).read_bytes()
        assert sha(row['raw_file'])==row['raw_sha256'] and len(body)==row['raw_bytes']<=LIMIT
        assert row['original_metadata_bytes']==int(story['story_file_size'])
        assert abs(row['raw_size_relative_change']-(len(body)/int(story['story_file_size'])-1))<1e-12
        attempt=row['attempts'][-1]
        try:text,encoding=visible_text(body,attempt['charset'],attempt['content_type'])
        except UnicodeDecodeError:
            assert row['usable'] is False and 'UnicodeDecodeError' in row['error']
        else:
            assert sha(row['text_file'])==row['text_sha256']
            assert Path(row['text_file']).read_bytes().decode('utf-8')==text and encoding==row['decoded_encoding']
            first,last,words,ordered=marker_positions(text,story)
            assert (first,last,words,ordered)==(row['start_marker_offset'],row['end_marker_offset'],row['normalized_words'],row['ordered_official_markers'])
            assert row['usable']==(ordered and words>=5000)
    else:
        assert row['usable'] is False and all('error' in attempt for attempt in row['attempts'])
    if row['usable']:
        usable.append(dict(story_id=row['story_id'],kind=row['kind'],context_tokens=row['context_tokens'],normalized_words=row['normalized_words'],raw_size_relative_change=row['raw_size_relative_change']))
    else:failures.append(dict(story_id=row['story_id'],kind=row['kind'],error=row['error']))
assert len(usable)==r['usable_stories']==14
assert sum(row['kind']=='gutenberg' for row in usable)==10
report=dict(selection_download_bytes_decode_text_markers_and_coverage_verified=True,
    manifest_sha256=sha(root/'manifest.json'),preflight_sha256=sha(root/'preflight.json'),
    selected_stories=20,usable_stories=14,books=10,movies=4,usable=usable,failures=failures,
    model_evaluation_started=False,
    limitations='Six explicit failures/no replacement. Availability/content predicates only; some contexts exceed31500tokens and will need recorded standard truncation. Header/content charset conflicts need separately preserved diagnosis; no independent narrative quality, complete semantic integrity or training-overlap claim.')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:report[key] for key in ('selected_stories','usable_stories','books','movies','failures')},indent=2))
