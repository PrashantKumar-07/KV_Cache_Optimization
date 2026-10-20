"""Fresh five-method context scaling; fixed teacher input, complete model+policy."""
import functools,gc,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from latency_profile import profile_one
from longbench_eval import _attn_impl,prefill_logit_kwargs
from evaluate_query_selection import PromptQueryCapture
from provenance import atomic_json,model_artifact_identity,runtime_identity
spec=importlib.util.spec_from_file_location('frozen_full',ROOT/'results/full_evaluation_20261015/run_full.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def hooks(m):return [(len(x._forward_pre_hooks),len(x._forward_hooks)) for x in m.modules()]
def quiet():
 uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
 others=[r for r in apps if r.split(',')[0].strip()==uuid and int(r.split(',')[1])!=os.getpid()]
 return {'gpu':1,'uuid':uuid,'other_compute_processes':others,'quiet':not others}
def instrument(model,arm):
 original=model.forward
 @functools.wraps(original)
 def forward(*args,**kwargs):
  if arm in ('full','streamingllm'):
   kwargs['output_attentions']=False
   with _attn_impl(model,'sdpa'):out=original(*args,**kwargs)
  else:out=original(*args,**kwargs)
  if not torch.isfinite(out.logits).all().item():raise FloatingPointError('Nonfinite logits')
  return out
 model.forward=forward
 return original
def main():
 plan=json.loads((BASE/'plan.json').read_text());outpath=BASE/'timing.json';assert not outpath.exists()
 assert sha(Path(__file__))==plan['runner_sha256']
 assert all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items())
 assert runtime_identity()==plan['runtime']
 assert model_artifact_identity(plan['model'])==plan['model_identity']
 for p,h in plan['inputs_sha256'].items():assert sha(ROOT/p)==h
 report={'execution_complete':False,'plan_sha256':sha(BASE/'plan.json'),'rows':[],'quiet_observations':[]}
 def check(stage):
  q=quiet();report['quiet_observations'].append({'stage':stage,**q});assert q['quiet'],'GPU1 contention; preserve incomplete run'
 try:
  check('before_model_load');torch.manual_seed(0)
  with torch.inference_mode():
   model=AutoModelForCausalLM.from_pretrained(plan['model'],torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
   tokenizer=AutoTokenizer.from_pretrained(plan['model'])
   probe=model(torch.tensor([[1]],device='cuda'),use_cache=False,output_attentions=True,**prefill_logit_kwargs(model,True));del probe
   text=(ROOT/plan['prompt']).read_text();teachers=json.loads((ROOT/plan['teachers']).read_text())['config']['teacher_token_ids']
   for ctx in plan['contexts']:
    for block in range(3):
     shift=block%5;order=plan['arms'][shift:]+plan['arms'][:shift]
     if block%2:order=order[::-1]
     for arm in order:
      check('before_row');before=hooks(model);policy=full.policy_for(arm)
      with ExitStack() as cleanup:
       cleanup.callback(policy.reset)
       if arm=='tiered_hot1024':
        capture=PromptQueryCapture(model,policy);cleanup.callback(capture.close)
       original=instrument(model,arm);cleanup.callback(setattr,model,'forward',original)
       row=profile_one(model,tokenizer,policy,ctx,n_decode=50,n_warmup=1,n_repeat=1,device='cuda',prompt_text=text,forced_token_ids=teachers,prefill_last_logits=True)
      assert hooks(model)==before
      del policy;gc.collect();torch.cuda.empty_cache();check('after_row')
      report['rows'].append({'context':ctx,'block':block,'arm':arm,'hooks_restored':True,'all_logits_finite':True,**row})
      atomic_json(outpath,report);print(ctx,block,arm,row['itl_ms'],flush=True)
   report['execution_complete']=True
 except BaseException as error:
  report['error']=f'{type(error).__name__}: {error}';raise
 finally:atomic_json(outpath,report)
if __name__=='__main__':main()
