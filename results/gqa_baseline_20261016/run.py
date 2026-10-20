"""Complete current KV-head SnapKV baseline under the frozen quality protocol."""
from pathlib import Path
import fcntl,gc,hashlib,importlib.util,json,os,sys,traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'results/improvement_design_20261016'),str(ROOT/'experiments'),str(ROOT/'src')]
from gqa_snap_adapter import GQASnapKVAdapter
from provenance import atomic_json,model_artifact_identity,runtime_identity,read_jsonl
spec=importlib.util.spec_from_file_location('frozen_generator',ROOT/'results/full_evaluation_20261015/run_full.py');full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
torch=full.torch
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 plan=json.loads((BASE/'plan.json').read_text());ph=sha(BASE/'plan.json')
 assert all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items())
 assert runtime_identity()==plan['runtime'];assert model_artifact_identity(plan['model'])==plan['model_identity']
 preflight=json.loads((ROOT/plan['preflight']).read_text());assert preflight['verified'] and preflight['execution_complete']
 assert preflight['plan_sha256']==plan['preflight_plan_sha256']
 directory=BASE/'records';directory.mkdir(exist_ok=True)
 with (directory/'.running.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
  torch.manual_seed(0)
  with torch.inference_mode():
   model=full.AutoModelForCausalLM.from_pretrained(plan['model'],torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
   probe=model(torch.tensor([[1]],device='cuda'),use_cache=False,output_attentions=True,**full.bench.prefill_logit_kwargs(model,True));del probe
   tokenizer=full.AutoTokenizer.from_pretrained(plan['model']);count=0
   atomic_json(directory/'runtime.json',{'runtime':runtime_identity(),'plan_sha256':ph,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device':torch.cuda.get_device_name()})
   for task in plan['tasks']:
    source=ROOT/plan['dataset_files'][task];assert sha(source)==plan['dataset_sha256'][task]
    for index,item in enumerate(read_jsonl(source)):
     sid=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest();assert sid==plan['sample_ids'][task][index]
     refp=ROOT/plan['reference_records'][task][index]['path'];assert sha(refp)==plan['reference_records'][task][index]['sha256'];ref=json.loads(refp.read_text())
     prompt=full.bench.DATASET_TO_PROMPT[task].format(**item)
     if task not in full.bench.NO_CHAT_TASKS:prompt='[INST] '+prompt+' [/INST]'
     ids=tokenizer(prompt,return_tensors='pt').input_ids;untruncated=ids.shape[1]
     if ids.shape[1]>31500:ids=torch.cat((ids[:,:15750],ids[:,-15750:]),1)
     th=hashlib.sha256(ids.numpy().tobytes()).hexdigest();assert th==ref['input_ids_sha256']
     p=directory/f'{task}_{index:04d}.json'
     identity={'plan_sha256':ph,'task':task,'physical_file_row_index':index,'sample_id':sid,'arm':'snapkv_current_kv_head','input_ids_sha256':th,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'untruncated_prompt_tokens':untruncated,'dataset_length':item.get('length'),'metric_function':full.bench.DATASET_TO_METRIC[task].__name__,'requested_backends':{'prefill':'sdpa','decode':'eager'}}
     if p.exists():
      previous=json.loads(p.read_text());assert previous['execution_complete'] and all(previous[k]==v for k,v in identity.items());count+=1;continue
     original=full.SnapKVAdapter
     try:
      full.SnapKVAdapter=GQASnapKVAdapter
      result=full.run_request(model,tokenizer,ids.to('cuda'),task,item,'snapkv_faithful')
      assert result['generated_token_ids'][0]==ref['generated_token_ids'][0]
      assert all(l['visible_k']['shape'][1]==8 for l in result['geometry_after_prefill_before_first_decode']['layers'])
      assert not p.exists();atomic_json(p,{**identity,**result,'execution_complete':True});count+=1;print(task,index,result['score'],flush=True)
     except BaseException:
      stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');atomic_json(BASE/'errors'/f'{task}_{index}_{stamp}.json',{**identity,'traceback':traceback.format_exc()});raise
     finally:full.SnapKVAdapter=original;gc.collect();torch.cuda.empty_cache()
   assert count==1474;atomic_json(directory/'complete.json',{'plan_sha256':ph,'execution_complete':True,'records':count})
if __name__=='__main__':main()
