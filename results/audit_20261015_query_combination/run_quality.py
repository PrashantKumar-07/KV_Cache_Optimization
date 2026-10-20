"""Prospective consumed-development ablation: actual prompt queries + decode retrieval.

All arms use the same frozen records, model, prompts and token identities.
This is an algorithmic GPU quality screen; it is not native-CPU deployment timing.
"""
from contextlib import ExitStack
from pathlib import Path
import gc,hashlib,json,sys
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import longbench_eval as bench
from evaluate_query_selection import PromptQueryCapture,QuerySelector
from snapkv_adapter import SnapKVAdapter
from provenance import atomic_json,model_artifact_identity,runtime_identity,read_jsonl
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((BASE/'plan.json').read_text());output=BASE/'quality.json';assert not output.exists()
assert sha(Path(__file__))==plan['runner_sha256']
assert all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items())
assert model_artifact_identity(plan['model'])==plan['model_identity']
for t,p in plan['dataset_files'].items():assert sha(ROOT/p)==plan['dataset_sha256'][t]
report={'execution_complete':False,'plan_sha256':sha(BASE/'plan.json'),'runtime':runtime_identity(),'rows':[]}
def hook_state(model):
 return [(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()],[l.self_attn.num_key_value_groups for l in model.model.layers]
class ObservedSelector(QuerySelector):
 def __init__(self,*args,**kwargs):
  self.observed={'eligible_layer_calls':0,'ranking_calls':0,'small_live_noops':0,'changed_position_selections':0}
  super().__init__(*args,**kwargs)
 def _hook(self,index):
  original=super()._hook(index)
  def observe(module,args,kwargs):
   hidden=kwargs.get('hidden_states',args[0] if args else None)
   old=list(self.policy._vis_stt_pos.get(index,[]))
   if hidden is not None and hidden.size(1)==1 and index in self.policy.layer_caches and old:
    self.observed['eligible_layer_calls']+=1
    live=int((~self.policy.layer_caches[index].stt_shadow).sum().item())
    self.observed['ranking_calls' if live>len(old) else 'small_live_noops']+=1
   result=original(module,args,kwargs)
   if list(self.policy._vis_stt_pos.get(index,[]))!=old:self.observed['changed_position_selections']+=1
   return result
  return observe
try:
 torch.manual_seed(0)
 with torch.inference_mode():
  model=AutoModelForCausalLM.from_pretrained(plan['model'],torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
  framework_probe=model(torch.tensor([[1]],device='cuda'),use_cache=False,output_attentions=True,logits_to_keep=1)
  del framework_probe
  tokenizer=AutoTokenizer.from_pretrained(plan['model'])
  for task in plan['tasks']:
   lookup={hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest():r for r in read_jsonl(ROOT/plan['dataset_files'][task])}
   for index,sid in enumerate(plan['sample_ids'][task]):
    item=lookup[sid];prompt=bench.DATASET_TO_PROMPT[task].format(**item)
    if task not in bench.NO_CHAT_TASKS:prompt='[INST] '+prompt+' [/INST]'
    ids=tokenizer(prompt,return_tensors='pt').input_ids
    if ids.shape[1]>31500:ids=torch.cat([ids[:,:15750],ids[:,-15750:]],1)
    tokenhash=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    assert tokenhash==plan['input_ids_sha256'][task][index]
    ids=ids.to('cuda');n=ids.shape[1];row={'task':task,'sample_id':sid,'input_ids_sha256':tokenhash,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'arms':{}}
    report['rows'].append(row)
    for arm in plan['order'][str(index%2)]:
     before=hook_state(model);policy=None
     try:
      with ExitStack() as cleanup:
       if arm=='snap':
        past=SnapKVAdapter(model,904,32,reuse_projection=True);cleanup.callback(past.close)
        with bench._attn_impl(model,'sdpa'):
         out=model(ids,past_key_values=past,use_cache=True,position_ids=torch.arange(n,device='cuda').unsqueeze(0),**bench.prefill_logit_kwargs(model,True))
       else:
        policy=bench.TieredKVPolicy(sram_budget=1024,stt_budget=2048,stt_expose_quota=32,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
        cleanup.callback(policy.reset)
        capture=PromptQueryCapture(model,policy);cleanup.callback(capture.close)
        if arm=='query':
         selector=ObservedSelector(model,policy);cleanup.callback(selector.close)
        with bench._attn_impl(model,'sdpa'):
         out=model(ids,use_cache=True,position_ids=torch.arange(n,device='cuda').unsqueeze(0),**bench.prefill_logit_kwargs(model,True))
        past=out.past_key_values;bench.apply_tuple_kv_inplace(past,policy(bench.to_tuple_kv(past)))
       generated=[]
       for step in range(bench.DATASET_TO_MAXGEN[task]):
        assert torch.isfinite(out.logits).all()
        token=out.logits[:,-1:].argmax(-1);tid=token.item();generated.append(tid);del out
        if tid==tokenizer.eos_token_id or step+1==bench.DATASET_TO_MAXGEN[task]:break
        with bench._attn_impl(model,'sdpa' if arm=='snap' else 'eager'):
         out=model(token,past_key_values=past,use_cache=True,position_ids=torch.tensor([[n+step]],device='cuda'),output_attentions=policy is not None)
        if policy is not None:
         raw=out.past_key_values;tup=bench.to_tuple_kv(raw);policy.update_scores(tup,out.attentions)
         bench.apply_tuple_kv_inplace(raw,policy(tup));past=raw;del tup,raw
       text=tokenizer.decode(generated,skip_special_tokens=True);scored=text.lstrip('\n').split('\n')[0] if task in ('triviaqa','trec','samsum','lsht') else text
       score=100*max(bench.DATASET_TO_METRIC[task](scored,a,all_classes=item.get('all_classes',[])) for a in item['answers'])
       row['arms'][arm]={'prediction':text,'score':score,'generated_token_ids':generated,'prompt_tokens':n,
        'cache_stats':None if policy is None else policy.stats()}
       if arm=='query':row['arms'][arm]['selector_observations']=dict(selector.observed)
       if arm in ('lru','snap'):
        assert text==plan['reference_predictions'][task][arm][index], 'Unchanged control prediction differs from prior frozen screen'
       del past,token
     finally:
      if policy is not None:policy.reset()
     assert hook_state(model)==before
     row['arms'][arm]['hooks_and_groups_restored']=True
     gc.collect();torch.cuda.empty_cache();atomic_json(output,report)
     print(task,index,arm,round(score,4),flush=True)
    del ids
  report['execution_complete']=True
except BaseException as error:
 report['error']=f'{type(error).__name__}: {error}';raise
finally:atomic_json(output,report)
