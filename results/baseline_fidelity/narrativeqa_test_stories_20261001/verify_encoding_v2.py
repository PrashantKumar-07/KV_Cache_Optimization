"""Verify versioned charset corrections without source or selection replacement."""
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path[:0]=['experiments','src']
from download_narrativeqa_test_stories import visible_text,marker_positions
root=Path('results/baseline_fidelity/narrativeqa_test_stories_20261001')
output=root/'verified_encoding_v2.json';assert not output.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((root/'manifest.json').read_text());old=json.loads((root/'preflight.json').read_text());r=json.loads((root/'encoding_preflight_v2.json').read_text())
assert r['preflight_completed'] and r['original_preflight_sha256']==sha(root/'preflight.json')
assert r['manifest_sha256']==sha(root/'manifest.json')
assert r['runner_sha256']==sha('experiments/redecode_narrativeqa_stories.py')
assert r['html_observer_sha256']==sha('experiments/download_narrativeqa_test_stories.py')
labels_path=Path('/data/nishant/tieredkv_external/narrativeqa_metadata_20261001/encoding_labels_20261001.json')
assert r['encoding_labels_sha256']==sha(labels_path)
labels=json.loads(labels_path.read_text())
assert any(entry['name']=='windows-1252' and 'iso-8859-1' in entry['labels'] for section in labels for entry in section['encodings'])
assert len(r['rows'])==len(old['rows'])==20
stories={row['story']['document_id']:row['story'] for row in m['selected']};repaired=[];failures=[]
for before,after in zip(old['rows'],r['rows']):
    assert before['story_id']==after['story_id'] and before['attempts']==after['attempts']
    if 'UnicodeDecodeError' not in before.get('error',''):
        assert before==after
    else:
        assert after['original_error']==before['error']
        assert before['raw_file']==after['raw_file'] and before['raw_sha256']==after['raw_sha256']==sha(after['raw_file'])
        body=Path(after['raw_file']).read_bytes()
        assert re.search(rb'<meta\b[^>]*charset\s*=\s*iso-8859-1',body[:4096],re.I)
        assert body.decode('cp1252').encode('cp1252')==body
        text,encoding=visible_text(body,'cp1252',after['attempts'][-1]['content_type'])
        assert encoding==after['decoded_encoding']=='cp1252'
        assert Path(after['text_file']).read_bytes().decode('utf8')==text and sha(after['text_file'])==after['text_sha256']
        first,last,words,ordered=marker_positions(text,stories[after['story_id']])
        assert (first,last,words,ordered)==(after['start_marker_offset'],after['end_marker_offset'],after['normalized_words'],after['ordered_official_markers'])
        assert after['usable']==(ordered and words>=5000)
        assert after['encoding_adapter']['raw_byte_roundtrip'] and after['encoding_adapter']['network_requests']==0
        repaired.append(dict(story_id=after['story_id'],usable=after['usable']))
    if not after['usable']:failures.append(dict(story_id=after['story_id'],error=after['error']))
assert len(repaired)==3 and sum(row['usable'] for row in r['rows'])==r['usable_stories']==16
report=dict(source_selection_coverage_encoding_roundtrip_and_unchanged_gates_verified=True,
    manifest_sha256=sha(root/'manifest.json'),original_preflight_sha256=sha(root/'preflight.json'),encoding_preflight_sha256=sha(root/'encoding_preflight_v2.json'),
    selected_stories=20,usable_stories=16,books=10,movies=6,charset_repairs=repaired,failures=failures,
    narrative_model_evaluation_started=False,
    limitations='Four original selections still fail/no replacements. Three missing start markers refer to original Javascript boilerplate if(window), not visible story text; strict gate remains failed pending fidelity review. Current adapted source/text, standard truncation, semantic integrity and training overlap qualifications remain.')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
