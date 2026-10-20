"""Bounded original-story downloads for frozen NarrativeQA content preflight."""
import argparse
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.parse import urlsplit,urlunsplit
from urllib.request import Request,urlopen
from transformers import AutoTokenizer
from audit_narrativeqa_story_ids import normalized_tokens
from provenance import atomic_json

LIMIT=5*1024*1024


class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts=[];self.skip=0

    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):self.skip+=1
        if not self.skip and tag in ('p','br','div','pre','tr','li','h1','h2','h3'):self.parts.append('\n')

    def handle_endtag(self,tag):
        if tag in ('script','style') and self.skip:self.skip-=1
        if not self.skip and tag in ('p','div','pre','tr','li','h1','h2','h3'):self.parts.append('\n')

    def handle_data(self,data):
        if not self.skip:self.parts.append(data)


def visible_text(body,charset,content_type):
    if body.startswith(b'\x1f\x8b'):
        from io import BytesIO
        with gzip.GzipFile(fileobj=BytesIO(body)) as stream:body=stream.read(LIMIT+1)
        if len(body)>LIMIT:raise ValueError('Decompressed content exceeds5MiB')
    encoding=charset or 'utf-8'
    try:text=body.decode(encoding)
    except UnicodeDecodeError:
        if charset:raise
        encoding='cp1252';text=body.decode(encoding)
    if 'html' in content_type.casefold() or '<html' in text[:1024].casefold() or '<!doctype html' in text[:1024].casefold():
        parser=VisibleHTML();parser.feed(text);parser.close();text=''.join(parser.parts)
    return text,encoding


def marker_positions(text,story):
    normalized=normalized_tokens(text)
    start=normalized_tokens(story['story_start']);end=normalized_tokens(story['story_end'])
    assert start and end
    padded = " " + normalized + " "
    first=padded.find(" " + start + " ")
    last=padded.rfind(" " + end + " ")
    return first,last,len(normalized.split()),first>=0 and last>=first+len(start)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve prior evidence: {args.output}')
    m=json.loads(args.manifest.read_text());assert m['passed'] and len(m['selected'])==20
    report=dict(preflight_completed=False,rows=[],manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observation_dependency_sha256=hashlib.sha256(Path('experiments/audit_narrativeqa_story_ids.py').read_bytes()).hexdigest())
    rawdir=args.output.parent/'raw';textdir=args.output.parent/'text'
    rawdir.mkdir(parents=True,exist_ok=True);textdir.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct',local_files_only=True)
    try:
        for selected in m['selected']:
            story=selected['story'];identifier=story['document_id']
            rawpath=rawdir/f'{identifier}.body';textpath=textdir/f'{identifier}.txt'
            assert not rawpath.exists() and not textpath.exists(),'Preserve prior story bytes'
            row=dict(story_id=identifier,kind=story['kind'],usable=False,attempts=[])
            try:
                original=story['story_url'];split=urlsplit(original)
                assert split.scheme in ('http','https')
                urls=[original]
                if split.scheme=='http':urls.append(urlunsplit(split._replace(scheme='https')))
                body=None
                for url in urls:
                    attempt=dict(requested_url=url);row['attempts'].append(attempt)
                    if story['kind']=='gutenberg':time.sleep(2)
                    try:
                        with urlopen(Request(url,headers={'User-Agent':'TieredKV-research-data-preflight/1.0'}),timeout=20) as response:
                            attempt.update(final_url=response.geturl(),status=response.status,
                                content_type=response.headers.get('Content-Type',''),charset=response.headers.get_content_charset())
                            body=response.read(LIMIT+1)
                        if len(body)>LIMIT:
                            body=None;raise ValueError('Response exceeds5MiB; truncated response rejected')
                        break
                    except Exception as error:
                        body=None;attempt['error']=f'{type(error).__name__}: {error}'
                if body is None:raise RuntimeError('All recorded URL attempts failed')
                with rawpath.open('xb') as stream:stream.write(body)
                row.update(raw_file=str(rawpath),raw_sha256=hashlib.sha256(body).hexdigest(),raw_bytes=len(body),
                    original_metadata_bytes=int(story['story_file_size']),
                    raw_size_relative_change=len(body)/int(story['story_file_size'])-1)
                text,encoding=visible_text(body,attempt['charset'],attempt['content_type'])
                with textpath.open('x') as stream:stream.write(text)
                first,last,words,markers=marker_positions(text,story)
                row.update(text_file=str(textpath),text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    decoded_encoding=encoding,normalized_words=words,context_tokens=len(tokenizer.encode(text)),
                    start_marker_offset=first,end_marker_offset=last,ordered_official_markers=markers,
                    usable=markers and words>=5000)
                if not row['usable']:row['error']='Official ordered markers or5000-word minimum failed'
            except Exception as error:
                row['error']=f'{type(error).__name__}: {error}'
            report['rows'].append(row);atomic_json(args.output,report)
            print(identifier,story['kind'],'usable',row['usable'],row.get('error',''),flush=True)
        report.update(preflight_completed=True,usable_stories=sum(row['usable'] for row in report['rows']),
            limitations='Frozen20 stories/no replacements. Raw current source may drift from2017 byte size; marker/word checks are feasibility, not complete semantic integrity. No model quality run, controller/STT or dataset equivalence claim.')
    finally:
        atomic_json(args.output,report)


if __name__=='__main__':
    main()
