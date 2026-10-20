"""Independent rescoring of the complete current upstream KV-head baseline."""
from pathlib import Path
import hashlib,importlib.util,json,sys
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
spec=importlib.util.spec_from_file_location('frozen_verifier',ROOT/'results/full_evaluation_20261015/verify_results.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
sha=helper.sha
def main():
 plan=json.loads((BASE/'plan.json').read_text());ph=sha(BASE/'plan.json');directory=BASE/'records'
 expected={directory/f'{t}_{i:04d}.json' for t in plan['tasks'] for i in range(plan['counts'][t])}
 assert len(expected)==1474 and set(directory.glob('*_*.json'))==expected,'Full1474 required before aggregation'
 completion=json.loads((directory/'complete.json').read_text());assert completion['execution_complete'] and completion['records']==1474 and completion['plan_sha256']==ph
 assert not list((BASE/'errors').glob('*.json')),'Preserved errors require review before acceptance'
 assert all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items())
 from provenance import runtime_identity,model_artifact_identity,read_jsonl,atomic_json
 from transformers import AutoTokenizer
 assert runtime_identity()==plan['runtime'];assert model_artifact_identity(plan['model'])==plan['model_identity']
 assert json.loads((directory/'runtime.json').read_text())['runtime']==plan['runtime']
 metrics,reference=helper.reference_metrics();tokenizer=AutoTokenizer.from_pretrained(plan['model'])
 old=json.loads((ROOT/'results/full_evaluation_20261015/verified_results.json').read_text());assert old['verified'];oldrows={(r['task'],r['source_zero_based_index']):r for r in old['rows']}
 rows=[];hashes={}
 for task in plan['tasks']:
  source=ROOT/plan['dataset_files'][task];assert sha(source)==plan['dataset_sha256'][task]
  for i,item in enumerate(read_jsonl(source)):
   p=directory/f'{task}_{i:04d}.json';r=json.loads(p.read_text());hashes[str(p.relative_to(ROOT))]=sha(p);helper.assert_finite(r)
   assert r['plan_sha256']==ph and r['execution_complete'] and r['arm']=='snapkv_current_kv_head'
   sid=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest();assert r['sample_id']==sid==plan['sample_ids'][task][i]
   prompt=plan['protocol']['prompt_templates'][task].format(**item)
   if task not in plan['protocol']['no_chat_tasks']:prompt='[INST] '+prompt+' [/INST]'
   ids=tokenizer(prompt,return_tensors='np').input_ids
   if ids.shape[1]>31500:ids=np.concatenate((ids[:,:15750],ids[:,-15750:]),1)
   assert r['input_ids_sha256']==hashlib.sha256(ids.tobytes()).hexdigest();assert r['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
   refp=ROOT/plan['reference_records'][task][i]['path'];assert sha(refp)==plan['reference_records'][task][i]['sha256'];ref=json.loads(refp.read_text());assert r['generated_token_ids'][0]==ref['generated_token_ids'][0]
   generated=r['generated_token_ids'];assert len(generated)==r['generated_tokens'] and 1<=len(generated)<=plan['protocol']['max_gen'][task]
   assert tokenizer.eos_token_id not in generated[:-1];assert len(generated)==plan['protocol']['max_gen'][task] or generated[-1]==tokenizer.eos_token_id
   assert tokenizer.decode(generated,skip_special_tokens=True)==r['prediction']
   text=r['prediction'].lstrip('\n').split('\n')[0] if task=='triviaqa' else r['prediction'];score=100*max(metrics[plan['protocol']['metrics'][task]](text,a) for a in item['answers']);assert abs(score-r['score'])<1e-9
   assert r['all_model_logits_finite'] and r['hooks_and_groups_restored']
   helper.verify_geometry(r,'snapkv_faithful',r['prompt_tokens'],32,8,8,128)
   base=oldrows[task,i];rows.append({'task':task,'index':i,'sample_id':sid,'partition':base['partition'],'original135':base['original135'],'score':score,'full_run_scores':base['scores'],'generated_tokens':r['generated_tokens'],'initial_payload_bytes':r['geometry_after_prefill_before_first_decode']['unique_payload_backing_bytes'],'final_payload_bytes':r['geometry_after_last_model_call_and_policy_commit']['unique_payload_backing_bytes']})
 summary={}
 for task in plan['tasks']:
  rr=[r for r in rows if r['task']==task];mean=float(np.mean([r['score'] for r in rr]));summary[task]={'n':len(rr),'mean_current_kv_head_snap':mean,'tiered_hot1024_minus_current_snap':float(np.mean([r['full_run_scores']['tiered_hot1024']-r['score'] for r in rr])),'tiered_hot992_minus_current_snap':float(np.mean([r['full_run_scores']['tiered_hot992']-r['score'] for r in rr]))}
 out=BASE/'verified_results.json';assert not out.exists()
 atomic_json(out,{'verified':True,'execution_complete':True,'requests':1474,'plan_sha256':ph,'verifier_sha256':sha(Path(__file__)),'reference_metric_sha256':sha(reference),'raw_records_sha256':hashes,'task_summaries':summary,'equal_task_macro_current_snap':float(np.mean([s['mean_current_kv_head_snap'] for s in summary.values()])),'scope':'Descriptive additional official KV-head comparator on already-unblinded1474originalsource records. No tuning, independence, equal-physical-memory or globalSOTA inference. CurrentSnap capacity1024 grows throughdecode; Tieredfixedhot992/1024+cold2048 has different memory. Recordpayloads are endpointstorages, notwholememory/runtimepeaks.','rows':rows})
 print('PASS1474 independently rescored currentGQA Snap predictions')
if __name__=='__main__':main()
