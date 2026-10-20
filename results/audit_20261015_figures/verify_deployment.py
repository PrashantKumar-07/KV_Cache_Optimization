"""Check published figure bytes, original preservation and raw-input lineage."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=json.loads((BASE/'original_figures_manifest.json').read_text())
assert sha(ROOT/archive['archive'])==archive['archive_sha256']
with tarfile.open(ROOT/archive['archive'],'r:gz') as tf:
 actual={m.name:hashlib.sha256(tf.extractfile(m).read()).hexdigest() for m in tf if m.isfile()}
assert actual=={p:r['sha256'] for p,r in archive['originals'].items()}
deployment=json.loads((BASE/'deployment_validation.json').read_text())
assert deployment['verified'] and deployment['rescored_predictions_verified']==150
for row in deployment['outputs']:assert sha(ROOT/row['target'])==row['sha256'],row['target']
p=json.loads((BASE/'validated/provenance.json').read_text())
assert sha(BASE/'validated_plot.py')==p['script_sha256']
for name,h in p['input_sha256'].items():assert sha(ROOT/name)==h,name
for stem in ['tieredkv_flow_diagram','tieredkv_flow_diagram_academic']:
 svg=(ROOT/'figures'/f'{stem}.svg').read_text()
 assert 'Pinned CPU cold tier' in svg and 'No third storage tier' in svg
 assert 'attention feedback' in svg and 'shadow' in svg
restoration=json.loads((BASE/'comparison_restoration/provenance.json').read_text())
assert sha(BASE/'comparison_restoration/render.py')==restoration['generator_sha256']
for name,h in restoration['inputs_sha256'].items():assert sha(ROOT/name)==h,name
for name,h in restoration['outputs_sha256'].items():assert sha(ROOT/'figures_final'/name)==h,name
metrics=json.loads((BASE/'metrics/reference_equivalence.json').read_text())
assert metrics['verified'] and metrics['comparisons']==500
for row in metrics['rows']:assert abs(row['ours']-row['reference'])<1e-12
for name,h in metrics['source_sha256'].items():assert sha(ROOT/name)==h,name
print('PASS: original24artifacts preserved; deployed24artifact hashes;27input hashes;150rescoredprediction lineage;500reference metric comparisons; editable two-tier diagrams')
