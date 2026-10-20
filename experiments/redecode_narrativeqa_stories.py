"""Offline versioned decoding of preserved contradictory HTTP/HTML encodings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from transformers import AutoTokenizer
from download_narrativeqa_test_stories import visible_text,marker_positions
from provenance import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    m=json.loads(args.manifest.read_text());old=json.loads(args.preflight.read_text())
    assert old['preflight_completed'] and old['manifest_sha256']==hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    labels=json.loads(args.labels.read_text())
    latin=[entry for section in labels for entry in section['encodings'] if 'iso-8859-1' in entry['labels']]
    assert len(latin)==1 and latin[0]['name']=='windows-1252'
    report=dict(old)
    report.update(preflight_completed=False,original_preflight_sha256=hashlib.sha256(args.preflight.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        encoding_labels_sha256=hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        html_observer_sha256=hashlib.sha256(Path('experiments/download_narrativeqa_test_stories.py').read_bytes()).hexdigest(),
        charset_adapter_scope='Only original UTF8 errors with explicit in-body ISO8859-1; strict Windows1252+byte roundtrip, no network/replacement/changed marker gate')
    by_story={selected['story']['document_id']:selected['story'] for selected in m['selected']}
    tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct',local_files_only=True)
    outdir=args.output.parent/'text_v2';outdir.mkdir(exist_ok=True)
    candidates=[row for row in report['rows'] if 'UnicodeDecodeError' in row.get('error','')]
    assert len(candidates)==3
    try:
        for row in candidates:
            row['original_error']=row['error']
            try:
                body=Path(row['raw_file']).read_bytes()
                assert hashlib.sha256(body).hexdigest()==row['raw_sha256']
                declared=row['attempts'][-1]['charset']
                assert declared.casefold()=='utf-8'
                try:body.decode(declared)
                except UnicodeDecodeError:pass
                else:raise AssertionError('Original header decoding unexpectedly succeeds')
                meta=set(label.lower().decode('ascii') for label in re.findall(rb'<meta\b[^>]*charset\s*=\s*([a-zA-Z0-9_-]+)',body[:4096],re.I))
                assert meta=={'iso-8859-1'},'Absent/ambiguous body encoding label'
                decoded=body.decode('cp1252');assert decoded.encode('cp1252')==body
                text,encoding=visible_text(body,'cp1252',row['attempts'][-1]['content_type'])
                path=outdir/f"{row['story_id']}.txt";assert not path.exists()
                with path.open('x') as stream:stream.write(text)
                first,last,words,ordered=marker_positions(text,by_story[row['story_id']])
                row.update(text_file=str(path),text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    decoded_encoding=encoding,normalized_words=words,context_tokens=len(tokenizer.encode(text)),
                    start_marker_offset=first,end_marker_offset=last,ordered_official_markers=ordered,
                    usable=ordered and words>=5000,encoding_adapter=dict(body_meta_label='iso-8859-1',
                        chosen_encoding='cp1252',raw_byte_roundtrip=True,network_requests=0))
                if row['usable']:row.pop('error')
                else:row['error']='Original ordered marker/5000-word gate still failed after decoding'
            except Exception as error:row['error']=f'{type(error).__name__}: {error}'
            print(row['story_id'],'usable',row['usable'],row.get('error',''),flush=True)
        report.update(preflight_completed=True,usable_stories=sum(row['usable'] for row in report['rows']),
            limitations='Versioned offline charset adapter; no new fetch/story replacement. Original errors retained. Availability/marker/word gates only, no semantic/annotation/training-integrity or model quality claim.')
    finally:atomic_json(args.output,report)


if __name__=='__main__':
    main()
