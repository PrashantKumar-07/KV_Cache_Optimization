"""Rescore all preregistered arms and document-cluster the paired differences."""
from pathlib import Path
import datetime,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from longbench_metrics import DATASET_TO_METRIC
from audit_unused_documents import document_group_key
from provenance import read_jsonl,model_artifact_identity,runtime_identity
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
target=BASE/'verified_quality.json';assert not target.exists()
p=json.loads((BASE/'plan.json').read_text());r=json.loads((BASE/'quality.json').read_text())
assert r['execution_complete'] and 'error' not in r
assert r['plan_sha256']==sha(BASE/'plan.json') and sha(BASE/'run_quality.py')==p['runner_sha256']
assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
assert model_artifact_identity(p['model'])==p['model_identity'] and r['runtime']==runtime_identity()
assert len(r['rows'])==30
rng=np.random.default_rng(0);boot=[];per_task={};n=0;observed={}
for task in p['tasks']:
 source=ROOT/p['dataset_files'][task];assert sha(source)==p['dataset_sha256'][task]
 lookup={hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest():x for x in read_jsonl(source)}
 rows=[x for x in r['rows'] if x['task']==task];assert [x['sample_id'] for x in rows]==p['sample_ids'][task]
 scores={arm:[] for arm in ('lru','query','snap')};groups={}
 for i,row in enumerate(rows):
  assert row['input_ids_sha256']==p['input_ids_sha256'][task][i]
  item=lookup[row['sample_id']];groups.setdefault(document_group_key(item,task),[]).append(i)
  assert set(row['arms'])==set(scores)
  for arm in scores:
   a=row['arms'][arm];assert a['hooks_and_groups_restored']
   pred=a['prediction'];scored=pred.lstrip('\n').split('\n')[0] if task in ('triviaqa','trec','samsum','lsht') else pred
   score=100*max(DATASET_TO_METRIC[task](scored,ans,all_classes=item.get('all_classes',[])) for ans in item['answers'])
   assert abs(score-a['score'])<1e-10
   if arm in ('snap','lru'):assert pred==p['reference_predictions'][task][arm][i]
   scores[arm].append(score);n+=1
  activity=row['arms']['query']['selector_observations']
  assert activity['eligible_layer_calls']==activity['ranking_calls']+activity['small_live_noops']
  for k,v in activity.items():observed[k]=observed.get(k,0)+v
 indices=list(groups.values());sizes=np.array([len(g) for g in indices]);draw=rng.integers(0,len(indices),(10000,len(indices)))
 bs={arm:np.array([np.array(values)[g].sum() for g in indices])[draw].sum(1)/sizes[draw].sum(1) for arm,values in scores.items()}
 delta={arm:bs['query']-bs[arm] for arm in ('lru','snap')};boot.append(delta)
 per_task[task]={'rows':5,'target_document_groups':len(indices),'scores':scores,
  'means':{arm:float(np.mean(v)) for arm,v in scores.items()},
  'query_minus':{arm:{'mean':float(np.mean(scores['query'])-np.mean(scores[arm])),
   'document_cluster95':np.quantile(delta[arm],[.025,.975]).tolist()} for arm in ('lru','snap')}}
assert n==90 and observed['ranking_calls']>0 and observed['changed_position_selections']>0
comparisons={arm:{'mean':float(np.mean([row['query_minus'][arm]['mean'] for row in per_task.values()])),
 'within_task_target_document_cluster_bootstrap95':np.quantile(np.mean([x[arm] for x in boot],axis=0),[.025,.975]).tolist()} for arm in ('lru','snap')}
guards={'query_minus_lru_at_least_2pts':comparisons['lru']['mean']>=2,
 'query_minus_lru_cluster_lower_above_zero':comparisons['lru']['within_task_target_document_cluster_bootstrap95'][0]>0,
 'query_minus_snap_positive_point_estimate':comparisons['snap']['mean']>0}
out={'verified':True,'reviewer':'root executable raw-rescoring; independent follow-up pending',
 'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rescored_predictions':n,
 'comparisons':comparisons,'registered_quality_guards':guards,'advancement_passed':all(guards.values()),
 'decision':'advance algorithmic development quality only' if all(guards.values()) else 'REJECT combination expansion; preserve failure; no favorable retry',
 'per_task':per_task,'selector_observations':observed,'adopted':False,
 'scope':p['scope'],'statistics':p['statistics'],
 'evidence_sha256':{str(q.relative_to(ROOT)):sha(q) for q in [BASE/'plan.json',BASE/'run_quality.py',BASE/'quality.json',Path(__file__)]}}
target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('per_task','evidence_sha256')},indent=2))
