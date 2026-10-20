"""Independent raw rescoring for the registered faithful-selector boundary."""
from pathlib import Path
import datetime,hashlib,json,sys,tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments'))
import longbench_eval as bench
from audit_unused_documents import document_group_key
from provenance import model_artifact_identity,read_jsonl
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=BASE/'verified_quality.json';assert not out.exists()
plan=json.loads((BASE/'plan.json').read_text())
snap=json.loads((BASE/'snap.json').read_text());tiered=json.loads((BASE/'tiered.json').read_text())
assert snap['config']['runtime']==tiered['config']['runtime']
assert snap['config']['model_identity']==tiered['config']['model_identity']==model_artifact_identity(snap['config']['model'])
assert snap['config']['capacity']==904 and snap['config']['window']==32
assert tiered['config']['real_query_prefill'] and tiered['config']['strategy']=='lru'
assert tiered['config']['tiered_vram_budget']==1024 and tiered['config']['tiered_stt_budget']==2048
assert snap['config']['prefill_last_logits'] and tiered['config']['prefill_last_logits']
assert set(snap['results'])==set(tiered['results'])==set(plan['tasks'])
assert sha(ROOT/plan['source_archive'])==plan['source_archive_sha256']
with tarfile.open(ROOT/plan['source_archive']) as tf:
 frozen={m.name:tf.extractfile(m).read() for m in tf if m.isfile()}
assert {n:hashlib.sha256(b).hexdigest() for n,b in frozen.items()}==plan['source_sha256']
assert all(sha(ROOT/n)==v for n,v in plan['source_sha256'].items())
h=hashlib.sha256()
for name in sorted((n for n in frozen if Path(n).parts[0] in ('src','experiments','sota')),key=Path):
 h.update(name.encode());h.update(frozen[name])
assert h.hexdigest()==snap['config']['runtime']['source_sha256']
rng=np.random.default_rng(0);boots=[];tasks={};count=0
for task in plan['tasks']:
 source=bench.task_data_path(snap['config']['data_dir'],task)
 assert sha(source)==snap['config']['dataset_sha256'][task]==tiered['config']['dataset_sha256'][task]
 data=read_jsonl(source)[:5]
 arms=[snap['results'][task]['snapkv_upstream_selector'],tiered['results'][task]['tieredkv']]
 assert all(arm['n']==len(data)==len(arm['samples'])==5 for arm in arms)
 differences=[];groups={}
 for i,item in enumerate(data):
  rows=[arm['samples'][i] for arm in arms]
  for key in ('sample_id','prompt_sha256','input_ids_sha256','prompt_tokens','max_ctx'):
   assert rows[0][key]==rows[1][key],(task,i,key)
  assert rows[0]['sample_id']==hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
  prompt=bench.DATASET_TO_PROMPT[task].format(**item)
  if task not in bench.NO_CHAT_TASKS:prompt=f'[INST] {prompt} [/INST]'
  assert rows[0]['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
  for row in rows:
   prediction=row['prediction']
   if task in ('trec','triviaqa','samsum','lsht'):prediction=prediction.lstrip('\n').split('\n')[0]
   score=100*max(bench.DATASET_TO_METRIC[task](prediction,answer,all_classes=item.get('all_classes',[])) for answer in item['answers'])
   assert abs(score-row['score'])<1e-9;count+=1
  difference=rows[1]['score']-rows[0]['score'];differences.append(difference)
  groups.setdefault(document_group_key(item,task),[]).append(difference)
 clusters=list(groups.values());sums=np.array([sum(v) for v in clusters]);sizes=np.array([len(v) for v in clusters])
 draws=rng.integers(0,len(clusters),(10000,len(clusters)))
 boots.append(sums[draws].sum(axis=1)/sizes[draws].sum(axis=1))
 tasks[task]={'rows':5,'target_document_groups':len(clusters),'snap_mean':float(np.mean([r['score'] for r in arms[0]['samples']])),
    'tiered_mean':float(np.mean([r['score'] for r in arms[1]['samples']])),'paired_difference':float(np.mean(differences)),
    'snap_final_expanded_cache_payload_bytes':[r['final_kv_payload_bytes'] for r in arms[0]['samples']],
    'tiered_final_internal_payload_bytes_excludes_packed_HF_cache':[r['final_resident_kv_payload_bytes'] for r in arms[1]['samples']],
    'snap_generated_tokens':[r['generated_tokens'] for r in arms[0]['samples']],
    'tiered_generated_tokens':[r['generated_tokens'] for r in arms[1]['samples']]}
interval=np.quantile(np.mean(boots,axis=0),[.025,.975]).tolist()
report={'verified':True,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rescored_predictions':count,
    'task_balanced_difference':float(np.mean([v['paired_difference'] for v in tasks.values()])),
    'within_task_target_document_cluster_bootstrap_95':interval,'quality_advancement_screen_passed':interval[0]>0,
    'method':'10000draws seed0; within-task target-passage clusters, preserve all paired rows, fixed-task means equalweight',
    'per_task':tasks,'budget_boundary':plan['budget_boundary'],'adopted':False,
    'limits':['Consumed development rows, one model/seed; no independent generalization orSOTA claim.',
      'Current upstream selector port, expandedhead/growingdecode behavior preserved.',
      'Per-row final payload fields have different boundaries: Tiered excludes packed HF KV; do not directly divide them.',
      'Only candidate has prefill allocation measurement; missing baseline fullallocation/width trajectories prevents complete resource comparison.',
      'Parallel GPUs/output lengths differ; elapsed fields do not establish causal speed or write gain.',
      'Prompt/token identity paired; token hashes not independently retokenized by this verifier.'],
    'evidence_sha256':{p.name:sha(p) for p in [BASE/'plan.json',BASE/'snap.json',BASE/'tiered.json',BASE/'verified_prompt_payload.json',Path(__file__)]}}
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('per_task','evidence_sha256')},indent=2))
