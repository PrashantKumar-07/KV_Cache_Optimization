"""Full-model proof of the current upstream KV-head Snap selector bridge."""
from pathlib import Path
import hashlib,json,math,sys,gc
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE),str(ROOT/'experiments'),str(ROOT/'src')]
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
from gqa_snap_adapter import GQASnapKVAdapter
from longbench_eval import _attn_impl,prefill_logit_kwargs
from provenance import atomic_json,model_artifact_identity,runtime_identity
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def tensor_hash(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def independent_select(k,q,v):
 groups=q.shape[1]//k.shape[1];window=32;cap=1024
 scores=q[:,:,-window:]@k.repeat_interleave(groups,dim=1).transpose(-1,-2)/math.sqrt(k.shape[-1])
 n=k.shape[-2];mask=torch.arange(n,device=k.device)[None,:]>torch.arange(n-window,n,device=k.device)[:,None]
 scores=scores.masked_fill(mask[None,None],float('-inf'))
 scores=torch.softmax(scores.float(),-1).to(q.dtype)[:,:,:,:-window].sum(-2)
 scores=scores.reshape(k.shape[0],k.shape[1],groups,n-window).mean(2)
 pooled=torch.nn.functional.avg_pool1d(scores,5,stride=1,padding=2)
 indices=pooled.topk(cap-window,dim=-1).indices[:,:,:,None].expand(-1,-1,-1,k.shape[-1])
 return tuple(torch.cat((t[:,:,:-window].gather(2,indices),t[:,:,-window:]),2) for t in (k,v))
class IndependentCache(GQASnapKVAdapter):
 def update(self,k,v,index,*args,**kwargs):
  if self.get_seq_length(index)==0:
   q=self.queries.pop(index);compressed=(k,v) if q.shape[-2]<1024 else independent_select(k,q,v)
   DynamicCache.update(self,*compressed,index);return k,v
  return DynamicCache.update(self,k,v,index)
def hooks(m):return [(len(x._forward_pre_hooks),len(x._forward_hooks)) for x in m.modules()],[l.self_attn.num_key_value_groups for l in m.model.layers]
def main():
 plan=json.loads((BASE/'gqa_preflight_plan.json').read_text());output=BASE/'gqa_gpu_proof.json';assert not output.exists()
 for p,h in plan['sha256'].items():assert sha(ROOT/p)==h,p
 assert model_artifact_identity(plan['model'])==plan['model_identity'];assert runtime_identity()==plan['runtime']
 report={'execution_complete':False,'plan_sha256':sha(BASE/'gqa_preflight_plan.json'),'contexts':[]}
 try:
  torch.manual_seed(0)
  with torch.inference_mode():
   model=AutoModelForCausalLM.from_pretrained(plan['model'],torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
   probe=model(torch.tensor([[1]],device='cuda'),use_cache=False,output_attentions=True,**prefill_logit_kwargs(model,True));del probe
   tokenizer=AutoTokenizer.from_pretrained(plan['model']);allids=tokenizer((ROOT/plan['prompt']).read_text(),return_tensors='pt').input_ids
   teacher=json.loads((ROOT/plan['teacher']).read_text())['config']['teacher_token_ids']
   before=hooks(model)
   for length in plan['contexts']:
    ids=allids[:,:length].to('cuda');assert ids.shape[1]==length
    for backend in plan['backends']:
     with _attn_impl(model,'sdpa'):
      full=model(ids,use_cache=False,position_ids=torch.arange(length,device='cuda')[None],**prefill_logit_kwargs(model,True)).logits
     reference_prefill=tensor_hash(full);del full
     traces=[]
     for cls in (GQASnapKVAdapter,IndependentCache):
      cache=cls(model,1024,32,5);trace=[]
      try:
       for step in range(6):
        tokens=ids if step==0 else torch.tensor([[teacher[step-1]]],device='cuda')
        pos=torch.arange(length,device='cuda')[None] if step==0 else torch.tensor([[length+step-1]],device='cuda')
        with _attn_impl(model,'sdpa' if step==0 else backend):out=model(tokens,past_key_values=cache,use_cache=True,position_ids=pos,**prefill_logit_kwargs(model,True))
        assert torch.isfinite(out.logits).all().item();lh=tensor_hash(out.logits)
        if step==0:assert lh==reference_prefill,'Full prefill logits changed'
        state=[]
        for layer in cache.layers:
         assert layer.keys.shape[1]==8 and layer.keys.shape[-2]==min(length,1024)+step
         state.append({'shape':list(layer.keys.shape),'k_sha256':tensor_hash(layer.keys),'v_sha256':tensor_hash(layer.values)})
        assert not cache.queries and not cache.pending_positions
        trace.append({'step':step,'logits_sha256':lh,'argmax':out.logits[:,-1:].argmax(-1).item(),'layers':state});del out
      finally:cache.close()
      assert hooks(model)==before
      traces.append(trace);del cache;gc.collect();torch.cuda.empty_cache()
     assert traces[0]==traces[1],'Upstream bridge differs from independent GQA mean-group reference'
     report['contexts'].append({'tokens':length,'backend':backend,'input_sha256':tensor_hash(ids),'prefill_full_logits_exact':True,'independent_trace_exact':True,'hooks_groups_restored':True,'trace':traces[0]})
     atomic_json(output,report);print(length,backend,'PASS',flush=True)
   report.update(execution_complete=True,verified=True,scope='Mistral7B BF16 full-prefill/5decode equality to independently implemented KV-head mean selector; not legacy query-head equivalence, quality, latency or SOTA evidence')
 except BaseException as error:report['error']=f'{type(error).__name__}: {error}';raise
 finally:atomic_json(output,report)
if __name__=='__main__':main()
