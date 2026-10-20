"""Package oversized raw JSON for GitHub while retaining original files locally."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/evidence'
LIMIT = 40 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST / 'manifest.json'
    assert not manifest_path.exists(), 'Preserve the existing evidence manifest'
    files = sorted(p for p in (ROOT / 'results').rglob('*.json')
                   if p.stat().st_size >= LIMIT and 'site-packages' not in p.parts)
    entries = []
    for source in files:
        name = str(source.relative_to(ROOT))
        entry = {'path': name, 'bytes': source.stat().st_size, 'sha256': sha(source), 'parts': []}
        archive_id = hashlib.sha256(name.encode()).hexdigest()[:16]
        with tempfile.TemporaryDirectory(prefix='.package-', dir=DEST) as tmp:
            compressed = Path(tmp) / 'record.gz'
            with source.open('rb') as data, compressed.open('wb') as output:
                with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as stream:
                    shutil.copyfileobj(data, stream)
            with compressed.open('rb') as stream:
                number = 0
                while block := stream.read(LIMIT):
                    target = DEST / f'{archive_id}.gz.part{number:03d}'
                    assert not target.exists()
                    target.write_bytes(block)
                    entry['parts'].append({'path': str(target.relative_to(ROOT)), 'bytes': len(block), 'sha256': sha(target)})
                    number += 1
        assert sha(source) == entry['sha256'], 'Source changed while packaging'
        entries.append(entry)
        print(f'Packaged {name}: {entry["bytes"]} bytes into {len(entry["parts"])} part(s)', flush=True)
    manifest_path.write_text(json.dumps({'schema': 1, 'compression': 'gzip, deterministic mtime0', 'files': entries}, indent=2) + '\n')


if __name__ == '__main__':
    main()
