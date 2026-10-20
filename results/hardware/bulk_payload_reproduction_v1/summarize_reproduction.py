"""Two-run component report; paired-block descriptive uncertainty only."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;R=B.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[R/'results/hardware/bulk_payload_controller_v1',B];blocks=[];runs=[]
for number,base in enumerate(paths,1):
 proof=json.loads((base/'verified_cost.json').read_text());raw=json.loads((base/'cost.json').read_text())
 assert proof['verified'] and proof['bulk_payload_controller_wholeloop_primary_passed']
 runs.append(dict(run=number,**proof['aggregate'],gain_percent=100*proof['relative_median_improvement'],saved_ms=proof['median_decode_saved_ms']))
 for block in range(3):
  rows=raw['rows'][block*4:block*4+4];assert [x['arm'] for x in rows]==['control','query','query','control']
  blocks.append({arm:np.array([v for x in rows if x['arm']==arm for v in x['trials'][0]['step_ms']]) for arm in ['control','query']})
assert len(blocks)==6 and all(len(x[a])==100 for x in blocks for a in ['control','query'])
arrays={a:np.stack([b[a] for b in blocks]) for a in ['control','query']}
a,b=[np.median(arrays[arm]) for arm in ['control','query']]
rng=np.random.default_rng(20261003);bootstrap=[]
for _ in range(10000):
 selection=rng.integers(0,6,size=6);x,y=[np.median(arrays[arm][selection]) for arm in ['control','query']];bootstrap.append([100*(1-y/x),x-y])
ci=np.quantile(bootstrap,[.025,.975],axis=0)
report=dict(verified=True,both_separate_process_gates_passed=True,runs=runs,combined_descriptive_medians=dict(control_ms=float(a),candidate_ms=float(b),gain_percent=float(100*(1-b/a)),saved_ms=float(a-b)),posthoc_paired_ABBA_block_bootstrap=dict(blocks=6,resamples=10000,seed=20261003,gain_percent_95_percentile=ci[:,0].tolist(),saved_ms_95_percentile=ci[:,1].tolist()),measured_forwards_per_arm_across_two_runs=600,source_sha256={str(base/f):sha(base/f) for base in paths for f in ['verified_cost.json','cost.json','cost_plan.json']},limits='Same fixed prompt/model/teacher/seed and code in two separate processes, not a separate investigator or task population. Six paired ABBA blocks only. Posthoc percentile intervals quantify run/block variation, not generalization or an additional preregistered adoption gate. Resource evidence from original process, not remeasured. No independent quality, faithful baseline or physical-device claim.',full_goal_complete=False)
out=B/'reproduction_summary.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
with (B/'headline_component.csv').open('w',newline='') as f:
 names=['run','arm','median_decode_ms','p95_decode_ms','median_ttft_ms'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
 for r in runs:
  for arm in ['control','query']:w.writerow(dict(run=r['run'],arm=arm,**r[arm]))
lines=[r'\begin{tabular}{lrrrr}',r'\hline',r'Run & CPU phase median & Bulk median & Gain (\%) & Saved (ms) \\',r'\hline']
for r in runs:lines.append(f"{r['run']} & {r['control']['median_decode_ms']:.2f} & {r['query']['median_decode_ms']:.2f} & {r['gain_percent']:.2f} & {r['saved_ms']:.2f} "+r'\\')
lines.extend([r'\hline',r'\end{tabular}','% Same-control component timing only; not a faithful-baseline/quality claim.'])
(B/'headline_component.tex').write_text('\n'.join(lines)+'\n');print(json.dumps(report['posthoc_paired_ABBA_block_bootstrap'],indent=2))
