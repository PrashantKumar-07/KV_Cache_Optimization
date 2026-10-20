"""Frozen fullmodel projection-reuse equivalence; timing is invalid."""
from pathlib import Path
import gc,hashlib,json,sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/audit_20261013/snap_projection_reuse_gpu'
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from snapkv_adapter import SnapKVAdapter
from longbench_eval import _attn_impl,prefill_logit_kwargs
from provenance import atomic_json,model_artifact_identity,runtime_identity
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((BASE/'plan.json').read_text());assert not (BASE/'parity.json').exists()
assert sha(Path(__file__))==plan['runner_sha256']
assert all(sha(ROOT/n)==h for n,h in plan['source_sha256'].items())
assert model_artifact_identity(plan['model'])==plan['model_identity']
for n in ['prompt','teacher_stream']:assert sha(ROOT/plan[n])==plan[n+'_sha256']
report={'passed':False,'plan_sha256':sha(BASE/'plan.json'),'runtime':runtime_identity(),'arms':{}}
def h(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
try:
 with torch.inference_mode():
  torch.manual_seed(0)
  model=AutoModelForCausalLM.from_pretrained(plan['model'],torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
  tokenizer=AutoTokenizer.from_pretrained(plan['model']);ids=tokenizer((ROOT/plan['prompt']).read_text(),return_tensors='pt').input_ids
  assert ids.shape[1]>=16384;ids=ids[:,:16384].to('cuda')
  report['input_ids_sha256']=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
  stream=json.loads((ROOT/plan['teacher_stream']).read_text())['config']['teacher_token_ids'][:51]
  assert len(stream)==51
  for name,reuse in [('reference',False),('reused',True)]:
   counts={'q':0,'k':0};handles=[]
   def count(n):
    def hook(*args):counts[n]+=1
    return hook
   for layer in model.model.layers:
    handles.append(layer.self_attn.q_proj.register_forward_hook(count('q')))
    handles.append(layer.self_attn.k_proj.register_forward_hook(count('k')))
   cache=SnapKVAdapter(model,904,32,reuse_projection=reuse);logits={};trace=[]
   try:
    for step in range(51):
     if step==0:
      with _attn_impl(model,'sdpa'):
       out=model(ids,past_key_values=cache,use_cache=True,position_ids=torch.arange(16384,device='cuda').unsqueeze(0),**prefill_logit_kwargs(model,True))
     else:
      out=model(torch.tensor([[stream[step-1]]],device='cuda'),past_key_values=cache,use_cache=True,position_ids=torch.tensor([[16384+step-1]],device='cuda'))
     assert torch.isfinite(out.logits).all();logits[str(step)]=out.logits.float().cpu().numpy();del out
     trace.append([{'key_sha256':h(l.keys),'value_sha256':h(l.values),'shape':list(l.keys.shape)} for l in cache.layers])
     assert not cache.queries and not cache.pending_positions
    np.savez_compressed(BASE/f'{name}_logits.npz',**logits)
    report['arms'][name]={'KV_trace':trace,'projection_counts':counts,'full_vocab_arrays':len(logits)}
   finally:
    cache.close()
    for handle in handles:handle.remove()
   assert all(l.self_attn.num_key_value_groups==4 and not l.self_attn._forward_pre_hooks and not l.self_attn._forward_hooks and not l.self_attn.q_proj._forward_hooks and not l.self_attn.k_proj._forward_hooks for l in model.model.layers)
   report['arms'][name]['hooks_and_groups_restored']=True
   del cache,logits;gc.collect();torch.cuda.empty_cache();atomic_json(BASE/'parity.json',report)
  a,b=report['arms']['reference'],report['arms']['reused'];assert a['KV_trace']==b['KV_trace']
  assert a['projection_counts']==dict(q=52*32,k=52*32)
  assert b['projection_counts']==dict(q=51*32,k=51*32)
  with np.load(BASE/'reference_logits.npz') as a,np.load(BASE/'reused_logits.npz') as b:
   assert set(a.files)==set(b.files)=={str(i) for i in range(51)}
   assert all(a[k].tobytes()==b[k].tobytes() and np.isfinite(a[k]).all() for k in a.files)
  report['passed']=True
finally:atomic_json(BASE/'parity.json',report)
