import hashlib,json,sys,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
kind=sys.argv[1];archive=BASE/(kind+'_source.tar.gz');out=BASE/(kind+'_implementation.json');assert not archive.exists() and not out.exists()
files=sorted(p for folder in ['src','experiments','tests'] for p in (ROOT/folder).rglob('*') if p.suffix in ['.py','.cpp','.cu','.h','.hpp'])+sorted(BASE.glob('*.py'))
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname=str(p.relative_to(ROOT)))
out.write_text(json.dumps(dict(plan_sha256=sha(BASE/('plan.json' if kind=='gpu' else 'cost_plan.json')),archive_sha256=sha(archive),sources={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2)+'\n')
