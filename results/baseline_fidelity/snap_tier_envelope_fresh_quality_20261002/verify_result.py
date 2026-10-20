"""Independent frozen-source/data/exclusion/resource and paired quality audit."""
import hashlib
import json
from pathlib import Path
import random
import sys
import tarfile
import numpy as np

P=Path(__file__).resolve().parent
ROOT=P.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import longbench_eval as bench
from audit_unused_documents import document_key
from provenance import read_jsonl

sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
read=lambda name:json.loads((P/name).read_text())
m,r,source=read('manifest.json'),read('result.json'),read('source_manifest.json')
for name in ('experiments/longbench_eval.py','experiments/longbench_metrics.py',
             'experiments/audit_unused_documents.py','experiments/make_document_holdout.py'):
 assert sha(ROOT/name)==source['source_sha256'][name], 'Use archived metric/normalizer version: '+name
assert r['passed'] and len(r['rows'])==r['expected_rows']==200
assert r['manifest_sha256']==sha(P/'manifest.json')
assert sha(P/'source.tar.gz')==source['bundle_sha256']
with tarfile.open(P/'source.tar.gz') as archive:
 for name,expected in source['source_sha256'].items():
  assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected,name
 for name,expected in r['source_sha256'].items():
  assert source['source_sha256'][name]==expected,name
 runtime_files=sorted(name for name in source['source_sha256'] if name.startswith('src/') and name.endswith('.py'))+[
  f'experiments/{name}' for name in ('longbench_eval.py','longbench_metrics.py','latency_profile.py','ppl_eval.py','provenance.py')]
 digest=hashlib.sha256()
 for name in runtime_files:
  digest.update(name.encode());digest.update(archive.extractfile(name).read())
 assert digest.hexdigest()==r['runtime']['source_sha256']
inventory_path=ROOT/m['inventory_file'];assert sha(inventory_path)==m['inventory_sha256']
inventory=json.loads(inventory_path.read_text());used=set(inventory['excluded_normalized_context_keys'])
assert inventory['passed'] and inventory['tasks']['narrativeqa']['eligible_document_groups']==0
rng=random.Random(m['selection_seed']);seen=set();task_differences={};regressions=[];records=[];expected_order=[]
eos=json.loads((ROOT/m['model']/'config.json').read_text())['eos_token_id']
eos_ids=set(eos if isinstance(eos,list) else [eos])
for task,meta in m['tasks'].items():
 source_path=ROOT/meta['source_file'];data_path=ROOT/meta['slice_file']
 assert sha(source_path)==meta['source_sha256'] and sha(data_path)==meta['slice_sha256']
 source_rows=read_jsonl(source_path);data=read_jsonl(data_path)
 assert len(data)==20 and data==[source_rows[i] for i in meta['selected_zero_based_indices']]
 groups={key:indices for key,indices in inventory['tasks'][task]['eligible_groups'].items() if key not in used|seen}
 keys=sorted(rng.sample(sorted(groups),20));indices=sorted(rng.choice(groups[key]) for key in keys)
 assert indices==meta['selected_zero_based_indices']
 differences=[]
 for item,sample,key in zip(data,meta['selected_sample_ids'],meta['selected_normalized_context_sha256']):
  assert sample==hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
  assert key==document_key(item) and key not in used|seen;seen.add(key)
  index=len(records)
  rows=[row for row in r['rows'] if row['sample_id']==sample and row['task']==task]
  assert len(rows)==2
  pair={row['arm']:row for row in rows};assert set(pair)=={'snap','cpu_direct'}
  order=m['arms'] if index%2==0 else m['arms'][::-1]
  expected_order.extend((index,task,sample,arm) for arm in order)
  assert pair['snap']['prompt_sha256']==pair['cpu_direct']['prompt_sha256']
  assert pair['snap']['generation']['prompt_tokens']==pair['cpu_direct']['generation']['prompt_tokens']
  for arm,row in pair.items():
   assert row['request_index']==index and row['normalized_context_sha256']==key
   count=len(row['generated_token_ids']);maximum=bench.DATASET_TO_MAXGEN[task]
   assert 0<count<=row['max_gen']==maximum
   assert count==row['generation']['generated_tokens']
   assert row['termination']==('eos' if row['generated_token_ids'][-1] in eos_ids else 'cap')
   assert row['termination']=='eos' or count==maximum
   assert 0<row['generation']['prompt_tokens']<=31500
   assert sum(row['final_owned_kv_storage_bytes'].values())<=m['owned_kv_ceiling_bytes']
   assert row['max_owned_kv_storage_bytes']<=m['owned_kv_ceiling_bytes']
   prediction=row['prediction']
   if task in ('trec','triviaqa','samsum','lsht'):prediction=prediction.lstrip('\n').split('\n')[0]
   score=100*max(bench.DATASET_TO_METRIC[task](prediction,answer,all_classes=item.get('all_classes',[])) for answer in item['answers'])
   assert abs(score-row['score'])<1e-8
   if arm=='snap':
    assert row['snap_prompt_capacity']==1025-maximum
    final_rows=min(row['generation']['prompt_tokens'],row['snap_prompt_capacity'])+count-1
    assert row['final_owned_kv_storage_bytes']=={'cuda:0':final_rows*524288}
    assert row['max_actual_forward_rows']==final_rows<=1024
   else:
    assert row['boundaries']==row['trajectory_frames']==count
    assert row['counters']['decode_steps']==count-1 and row['max_next_forward_rows']<=1024
    assert row['reserved_host_tensor_bytes']==row['final_owned_kv_storage_bytes']['cpu']==2082*131072
    assert set(row['prefill_query_shapes'])=={str(i) for i in range(32)}
    assert all(shape==[32,min(128,row['generation']['prompt_tokens']),128] for shape in row['prefill_query_shapes'].values())
  difference=pair['cpu_direct']['score']-pair['snap']['score'];differences.append(difference)
  record=dict(task=task,sample_id=sample,snap_score=pair['snap']['score'],cpu_score=pair['cpu_direct']['score'],difference=difference,
   snap_generated_tokens=len(pair['snap']['generated_token_ids']),cpu_generated_tokens=len(pair['cpu_direct']['generated_token_ids']),
   snap_final_owned_kv_bytes=sum(pair['snap']['final_owned_kv_storage_bytes'].values()),
   cpu_final_owned_kv_bytes=sum(pair['cpu_direct']['final_owned_kv_storage_bytes'].values()))
  records.append(record)
  if difference<0:regressions.append(record)
 task_differences[task]=np.asarray(differences,dtype=np.float64)
assert len(seen)==100 and len(task_differences)==5
assert [(row['request_index'],row['task'],row['sample_id'],row['arm']) for row in r['rows']]==expected_order
rng=np.random.default_rng(m['bootstrap_seed'])
draws=np.mean(np.stack([rng.choice(d,size=(m['bootstrap_draws'],len(d)),replace=True).mean(axis=1) for d in task_differences.values()]),axis=0)
ci=np.quantile(draws,[.025,.975]).tolist()
quality=dict(comparison='directCPU actual-query minus faithfulSnapKV at shared resource ceilings',
 task_balanced_mean_difference=float(np.mean([d.mean() for d in task_differences.values()])),
 paired_document_bootstrap_ci95=ci,bootstrap_seed=m['bootstrap_seed'],bootstrap_draws=m['bootstrap_draws'],
 noninferiority_margin=-.5,noninferiority_passed=ci[0]>-.5,superiority_passed=ci[0]>0,
 per_task={task:dict(n=20,mean_difference=float(d.mean()),regressions=int((d<0).sum()),losses_over_10=int((d<-10).sum())) for task,d in task_differences.items()},
 regression_count=len(regressions),losses_over_10_points=sum(record['difference']<-10 for record in regressions),
 regressions=regressions,sample_differences=records)
out=dict(verified=True,documents=100,arms=200,raw_sha256=sha(P/'result.json'),manifest_sha256=sha(P/'manifest.json'),
 source_bundle_sha256=source['bundle_sha256'],quality=quality,
 physical_allocation_scope='512MiB gate covers live KV tensor storage, not allocator capacity. InstalledTorch2.5source infers512MiB active pinned block for260.25MiB root; root capacity plusGPUKV≈763.75MiB before other allocations. No host-reservation telemetry or equalphysicalmemory proof.',
 resource_width_scope='Single-token decode only; full prefill is not capped. CPU max_next_forward_rows is a conservative next-step bound even after EOS. Snap max_actual_forward_rows stores final compressed geometry; it is not prefill width and no decode forward occurs when generation stops after one token.',
 limitations=r['limitations'])
dest=P/'verified_report.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in quality.items() if k not in ('regressions','sample_differences')},indent=2))
