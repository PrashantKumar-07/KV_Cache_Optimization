"""Verify frozen timing scope and the prospectively registered resource gate."""
from pathlib import Path
import datetime, hashlib, json, sys, tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[3]; BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments'))
from provenance import model_artifact_identity
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
output=BASE/'verified_resources.json'
assert not output.exists(), 'Preserve prior proof'
plan=json.loads((BASE/'plan.json').read_text())
raw=json.loads((BASE/'timing.json').read_text())
observed=json.loads((BASE/'observations.json').read_text())
assert observed['passed'] and observed['plan_sha256']==sha(BASE/'plan.json')
assert len(observed['quiet_observations'])==25
assert all(row['quiet'] and not row['unexpected_compute_processes'] for row in observed['quiet_observations'])
assert sha(BASE/'run_timing.py')==plan['runner_sha256']
for key in ('prompt','teacher_stream','quiet_helper'):
    assert sha(ROOT/plan[key])==plan[key+'_sha256']
with tarfile.open(BASE/'source.tar.gz') as tf:
    frozen={m.name:tf.extractfile(m).read() for m in tf if m.isfile()}
assert {name:hashlib.sha256(data).hexdigest() for name,data in frozen.items()}==plan['source_sha256']
assert all(sha(ROOT/name)==digest for name,digest in plan['source_sha256'].items())
h=hashlib.sha256()
for name in sorted((name for name in frozen if Path(name).parts[0] in ('src','experiments','sota')),key=Path):
    h.update(name.encode()); h.update(frozen[name])
assert h.hexdigest()==raw['config']['runtime']['source_sha256']
assert raw['config']['model_identity']==model_artifact_identity(raw['config']['model'])
assert raw['config']['prefill_last_logits'] is True
assert raw['config']['order']==plan['settings']['order']
assert raw['config']['ctx_len']==16384 and raw['config']['decode']==50
assert raw['config']['prefill_strategy']=='lru' and raw['config']['compare_prefill']
assert raw['config']['tiered_vram_budget']==1024 and raw['config']['tiered_stt_budget']==2048
assert raw['config']['prompt_sha256']==plan['prompt_sha256']
assert len(raw['results'])==len(observed['rows'])==12
arms={'proxy':[],'actual':[]}
for index,arm in enumerate(plan['settings']['order']):
    row=raw['results'][f'{index}_{arm}']
    assert row==observed['rows'][index]
    assert len(row['trials'])==2
    for trial in row['trials']:
        assert len(trial['step_ms'])==len(trial['forward_ms'])==len(trial['predicted_token_ids'])==trial['decode_steps']==50
        assert all(np.isfinite(v) and v>0 for v in trial['step_ms'])
        arms[arm].append(trial)
summary={}
for arm,trials in arms.items():
    steps=[v for trial in trials for v in trial['step_ms']]
    assert len(steps)==600
    assert all(trial['predicted_token_ids']==trials[0]['predicted_token_ids'] for trial in trials)
    summary[arm]={'measured_decode_forwards':len(steps),'median_itl_ms':float(np.median(steps)),
        'p95_itl_ms':float(np.quantile(steps,.95)),'median_ttft_ms':float(np.median([t['ttft_ms'] for t in trials])),
        'peak_prefill_alloc_gb':max(t['prefill_peak_memory_gb'] for t in trials),
        'peak_decode_alloc_gb':max(t['decode_peak_memory_gb'] for t in trials)}
ratios={name:summary['actual'][metric]/summary['proxy'][metric] for name,metric in
    [('median_itl_ratio','median_itl_ms'),('p95_itl_ratio','p95_itl_ms'),('median_ttft_ratio','median_ttft_ms'),
     ('peak_prefill_alloc_ratio','peak_prefill_alloc_gb'),('peak_decode_alloc_ratio','peak_decode_alloc_gb')]}
guards={name:ratios[name.removesuffix('_max')]<=ceiling for name,ceiling in plan['primary_guards'].items()}
report={'verified':True,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'summary':summary,
    'ratios':ratios,'registered_guards':guards,'resource_advancement_passed':all(guards.values()),
    'decision':'advance resource screen only' if all(guards.values()) else 'reject resource advancement; preserve outcome',
    'scope':plan['scope'],'adopted':False,'source_runtime_sha256':h.hexdigest(),
    'evidence_sha256':{p.name:sha(p) for p in [BASE/'plan.json',BASE/'source.tar.gz',BASE/'run_timing.py',BASE/'timing.json',BASE/'observations.json',Path(__file__)]}}
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['evidence_sha256']},indent=2))
