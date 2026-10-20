"""Verify mandatory native/free-generation/model-isolation and CLI evidence."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

P=Path(__file__).resolve().parent
read=lambda name:json.loads((P/name).read_text())
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(prefix):
 m=read(prefix+'_manifest.json');path=P/(prefix+'_source.tar.gz')
 assert sha(path)==m['bundle_sha256']
 with tarfile.open(path) as a:
  sources={name:a.extractfile(name).read() for name in m['source_sha256']}
 for name,value in sources.items():assert hashlib.sha256(value).hexdigest()==m['source_sha256'][name],name
 return m,sources


sources={prefix:archive(prefix) for prefix in ('stress','wide','isolation','integration','model')}
for prefix in ('stress','wide','isolation'):
 assert sources[prefix][0]['source_sha256']['experiments/bounded_pinned_pool.py']==sources['model'][0]['source_sha256']['experiments/bounded_pinned_pool.py']
old=ast.parse(sources['model'][1]['experiments/bounded_pinned_pool.py'])
new=ast.parse(sources['integration'][1]['experiments/bounded_pinned_pool.py'])
ctor=next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name=='__init__' and any(a.arg=='max_decode_append' for a in n.args.kwonlyargs))
guards=[n for n in ctor.body if isinstance(n,ast.If) and 'Hard protection exceeds' in ast.unparse(n)]
assert len(guards)==1;ctor.body.remove(guards[0]);assert ast.dump(old)==ast.dump(new)


def identity(raw,prefix):
 for name,expected in raw['source_sha256'].items():assert expected==sources[prefix][0]['source_sha256'][name],name


wide=read('wide_synthetic.json');identity(wide,'wide')
assert wide['passed'] and len(wide['checks'])==36
assert max(row['transient_logical_peak'] for row in wide['checks'])==1154
assert all(row['exact_state_payload_counters'] and row['rejected_append_before_mutation'] for row in wide['checks'])
generation=read('generation.json');identity(generation,'stress')
manifest=read('generation_manifest.json')
root=P.parents[2]
for task,meta in manifest['tasks'].items():
 path=root/meta['source_file'];assert sha(path)==meta['source_sha256']
 rows=[json.loads(line) for line in path.read_text().split('\n') if line.strip()]
 seen=set();indices=[]
 for index,item in enumerate(rows[:10 if task=='gov_report' else 25]):
  key=' '.join(item['context'].casefold().split())
  if key not in seen:seen.add(key);indices.append(index)
  if len(indices)==2:break
 assert indices==meta['selected_zero_based_indices']
 assert [hashlib.sha256(json.dumps(rows[index],sort_keys=True).encode()).hexdigest() for index in indices]==meta['selected_sample_ids']
assert generation['passed'] and generation['manifest_sha256']==sha(P/'generation_manifest.json')
assert len(generation['rows'])==48 and len(manifest['tasks'])==6
expected=[];number=0
for seed in (0,1):
 for task,meta in manifest['tasks'].items():
  for index,sample in enumerate(meta['selected_sample_ids']):
   order=('direct','bounded') if (number+seed)%2==0 else ('bounded','direct')
   expected.extend((seed,task,sample,arm) for arm in order)
   pair={row['arm']:row for row in generation['rows'] if row['seed']==seed and row['task']==task and row['sample_id']==sample}
   assert set(pair)=={'direct','bounded'}
   for key in ('prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256',
               'state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes',
               'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes'):
    assert pair['direct'][key]==pair['bounded'][key],key
   assert pair['direct']['reserved_host_tensor_bytes']==272891904 and pair['bounded']['reserved_host_tensor_bytes']==151257088
   assert len(pair['bounded']['generated_token_ids'])==pair['bounded']['trajectory_frames']
   assert pair['bounded']['counters']['decode_steps']==pair['bounded']['trajectory_frames']-1
   number+=1
assert [(row['seed'],row['task'],row['sample_id'],row['arm']) for row in generation['rows']]==expected
pool=read('multi_pool.json');identity(pool,'isolation')
assert pool['passed'] and pool['bounded'] and len(pool['checks'])==4
assert sum(row['exact_read_observations'] for row in pool['checks'])==768
assert sum(row['overlap_flushes'] for row in pool['checks'])==48 and sum(row['queued_write_resets'] for row in pool['checks'])==4
interleave=read('interleaved.json');identity(interleave,'isolation')
assert interleave['passed'] and interleave['bounded_interleaved'] and len(interleave['rows'])==9
assert interleave['hot']==991 and interleave['cold']==1090
for name,frames in (('A',13),('B',25),('C',13)):
 rows=[row for row in interleave['rows'] if row['request']==name];assert len(rows)==3
 assert {row['mode'] for row in rows}=={'gpu','cpu','interleaved_cpu'}
 fields=('predicted_token_ids','state_sha256','every_boundary_sha256','history_sha256','counters')
 for row in rows:
  assert len(row['every_boundary_sha256'])==frames
  for key in fields:assert row[key]==rows[0][key],key
control,candidate=read('cli_control.json'),read('cli_candidate.json')
for raw,bounded in ((control,False),(candidate,True)):
 config=raw['config'];assert config['cpu_bounded_pool']==bounded
 assert config['cpu_batched_scores'] and config['cpu_direct_shared'] and config['real_query_prefill'] and config['strategy']=='lru'
 assert config['tiered_vram_budget']==991 and config['tiered_stt_budget']==1090
 assert config['selector_sha256']==sources['integration'][0]['source_sha256']['experiments/evaluate_query_selection.py']
assert control['config']['dataset_sha256']==candidate['config']['dataset_sha256']
a,b=control['results']['qasper']['tieredkv'],candidate['results']['qasper']['tieredkv']
assert a['n']==b['n']==1 and a['score']==b['score']
x,y=a['samples'][0],b['samples'][0]
for key in ('sample_id','prediction','score','prompt_tokens','generated_tokens','final_resident_kv_payload_bytes',
            'final_hot_kv_payload_bytes','final_cpu_kv_occupied_bytes','submitted_h2d_bytes','submitted_d2h_bytes'):
 assert x[key]==y[key],key
assert x['final_cpu_kv_reserved_bytes']==272891904 and y['final_cpu_kv_reserved_bytes']==151257088
for key in ('promoted','demoted','paid_writes','writes_saved','dropped','decode_steps'):
 assert a['cache_stats'][key]==b['cache_stats'][key],key
invalid=read('invalid_flags.json');assert len(invalid)==3
assert all(row['exit_code']==2 and row['preload_rejection_verified'] for row in invalid)
primary=read('verified_report.json');assert primary['primary_and_regression_gates_passed']
out=dict(mandatory_stress_and_cli_verified=True,native_cases=36,full_transient_slots_reached=1154,
 generation_pairs=24,tasks=6,seeds=[0,1],delayed_read_observations=768,overlap_flushes=48,queued_resets=4,
 interleaved_requests=3,model_boundary_frames=51,constructor_guard_only_ast_preservation=True,
 cli_matching_sample=True,invalid_preload_flag_cases=3,
 primary_report_sha256=sha(P/'verified_report.json'),
 evidence_sha256={name:sha(P/name) for name in ('wide_synthetic.json','generation.json','multi_pool.json',
     'interleaved.json','cli_control.json','cli_candidate.json','invalid_flags.json')},
 limitations='Host memory improvement only; unchanged GPU model peak and originalqualityfailures. Development freegeneration, greedyseeds, serializedPython dispatch/interleaveddefaultmodelstream; no serving/thread safety/throughput/physicalSTT/exactphysicalresourcequality superiority or broadcompletion proof.')
dest=P/'verified_stress_report.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
