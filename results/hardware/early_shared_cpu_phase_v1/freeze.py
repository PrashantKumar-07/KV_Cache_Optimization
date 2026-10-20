"""Freeze measured source before launching a research gate."""
import datetime,hashlib,json,sys,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
kind=sys.argv[1]
files=sorted(p for d in ['src','experiments','tests'] for p in (ROOT/d).rglob('*') if p.suffix in ['.py','.cpp','.cu','.h','.hpp'])
files+=sorted(BASE.glob('*.py'))
files.append(ROOT/'results/hardware/layerwise_prefill_placement_v1/candidate.py')
archive=BASE/(kind+'_source.tar.gz');output=BASE/(kind+'_implementation.json')
assert not archive.exists() and not output.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=str(p.relative_to(ROOT)))
output.write_text(json.dumps(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(BASE/(kind+'_plan.json' if kind!='gpu' else 'plan.json')),archive_sha256=sha(archive),sources={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2)+'\n')
