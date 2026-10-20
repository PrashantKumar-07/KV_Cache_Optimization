"""Independent raw samples -> CSV/artist median audit, without renderer imports."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((HERE/'provenance.json').read_text());contract=json.loads((HERE/'numeric_contract.json').read_text())
for path,digest in manifest['inputs_sha256'].items():assert sha(ROOT/path)==digest
for path,digest in manifest['outputs_sha256'].items():assert sha(HERE/path)==digest
assert sha(HERE/'numeric_contract.json')==manifest['numeric_contract_sha256']
assert sha(HERE/'values.csv')==manifest['csv_sha256']
raw=json.loads((BASE/'timing.json').read_text());proof=json.loads((BASE/'verified_timing.json').read_text())
assert proof['verified'] and raw['execution_complete']
assert proof['raw_sha256']==sha(BASE/'timing.json')==contract['raw_sha256']
samples=defaultdict(list);blocks=defaultdict(set)
for r in raw['rows']:
    key=r['arm'],r['context'];blocks[key].add(r['block'])
    assert r['timing_scope']=='end_to_end_model_policy_token' and r['all_logits_finite'] and r['hooks_restored']
    for trial in r['trials']:samples[key].extend(trial['step_ms'])
assert len(samples)==30 and all(len(v)==150 for v in samples.values())
assert all(v=={0,1,2} for v in blocks.values()) and len(raw['quiet_observations'])==181
medians={key:(sorted(values)[74]+sorted(values)[75])/2 for key,values in samples.items()}
assert len(contract['artists'])==5
for artist in contract['artists']:
    assert artist['x']==[512,1024,2048,4096,8192,16384]
    for ctx,number in zip(artist['x'],artist['y']):
        assert number==medians[artist['arm'],ctx]==proof['summaries'][str(int(ctx))][artist['arm']]['median_itl_ms']
with (HERE/'values.csv').open() as f:
    rows=list(csv.DictReader(f));assert len(rows)==30
    for r in rows:assert float(r['median_itl_ms'])==medians[r['arm'],int(r['context_tokens'])]
result={'verified':True,'GPU_used':False,'raw_measured_samples':sum(map(len,samples.values())),
    'plotted_medians':30,'line_artists_and_CSV_match_raw_order_statistics':True,
    'original_five_methods_and_context_order':True,'no_confidence_interval_claim':contract['no_uncertainty_intervals'],
    'verifier_sha256':sha(__file__),'provenance_sha256':sha(HERE/'provenance.json')}
(HERE/'numeric_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
