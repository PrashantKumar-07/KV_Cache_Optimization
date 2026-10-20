"""Restore oversized research records from checksummed Git-sized gzip parts."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'artifacts/evidence/manifest.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe evidence path: {name}')
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError(f'Evidence path escapes repository: {name}')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Check restored records without writing')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    for entry in manifest['files']:
        target = inside(entry['path'])
        if target.exists():
            if target.stat().st_size != entry['bytes'] or digest(target) != entry['sha256']:
                raise ValueError(f'Existing record differs; preserve/review it before restoring: {entry["path"]}')
            print(f'OK {entry["path"]}')
            continue
        if args.verify_only:
            raise FileNotFoundError(f'Missing restored record: {entry["path"]}')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Temporary files stay on the destination filesystem, not a small /tmp.
        with tempfile.TemporaryDirectory(prefix='.restore-', dir=target.parent) as temporary:
            compressed = Path(temporary) / 'record.gz'
            with compressed.open('wb') as stream:
                for part in entry['parts']:
                    path = inside(part['path'])
                    if digest(path) != part['sha256'] or path.stat().st_size != part['bytes']:
                        raise ValueError(f'Corrupt archive part: {part["path"]}')
                    with path.open('rb') as source:
                        shutil.copyfileobj(source, stream)
            restored = Path(temporary) / 'record'
            with gzip.open(compressed, 'rb') as source, restored.open('wb') as stream:
                shutil.copyfileobj(source, stream)
            if digest(restored) != entry['sha256'] or restored.stat().st_size != entry['bytes']:
                raise ValueError(f'Restored record checksum mismatch: {entry["path"]}')
            restored.replace(target)
        print(f'Restored {entry["path"]}')
    print(f'Verified {len(manifest["files"])} original research records')


if __name__ == '__main__':
    main()
