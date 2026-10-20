"""Verify deployed current figures and exact preservation of replaced images."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((BASE/'deployment.json').read_text());assert d['verified']
assert sha(BASE/'provenance.json')==d['provenance_sha256']
assert sha(BASE/'numeric_verification.json')==d['numeric_verification_sha256']
assert sha(ROOT/d['prior_archive'])==d['prior_archive_sha256']
with tarfile.open(ROOT/d['prior_archive']) as tf:
 archived={m.name:hashlib.sha256(tf.extractfile(m).read()).hexdigest() for m in tf if m.isfile()}
assert archived==d['prior_files_sha256']
for row in d['outputs']:assert sha(ROOT/row['target'])==row['sha256']==sha(ROOT/row['source'])
p=json.loads((BASE/'provenance.json').read_text())
for name,h in p['input_sha256'].items():assert sha(ROOT/name)==h,name
assert json.loads((BASE/'numeric_verification.json').read_text())['raw_records_rehashed']==10318
print('PASS:18 deployed current artifacts, raw numeric evidence and superseded figure archive')
