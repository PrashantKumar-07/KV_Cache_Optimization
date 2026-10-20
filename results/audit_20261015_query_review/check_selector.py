"""CPU-only independent expanded-head check of existing query selector."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import torch
from transformers import MistralConfig,MistralForCausalLM
from transformers.models.mistral.modeling_mistral import rotate_half
from evaluate_query_selection import QuerySelector,PromptQueryCapture
from longbench_eval import TieredKVPolicy,to_tuple_kv,apply_tuple_kv_inplace

def run(dtype):
 torch.manual_seed(113)
 model=MistralForCausalLM(MistralConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,sliding_window=None,attn_implementation='eager')).eval().to(dtype)
 p=TieredKVPolicy(sram_budget=8,stt_budget=16,start_size=1,recent_size=2,stt_expose_quota=2,kv_dtype=dtype,sttram_bifurcation_frac=.75,promotion_budget=2)
 original=[(len(l.self_attn._forward_pre_hooks),l.self_attn.num_key_value_groups) for l in model.model.layers]
 capture=PromptQueryCapture(model,p);selector=QuerySelector(model,p,candidate_pages=0,use_index=False)
 observations=[];handles=[]
 def validate(i):
  def hook(m,args,kw):
   hidden=kw['hidden_states']
   if hidden.shape[1]!=1 or i not in p.layer_caches:return
   c=p.layer_caches[i];live=torch.where(~c.stt_shadow)[0];n=len(p._vis_stt_pos[i]);fast=p._vis_sram_keep[i]
   assert len(live)>n
   cos,sin=kw['position_embeddings'];q=m.q_proj(hidden).view(1,1,4,8).transpose(1,2)
   q=(q*cos.unsqueeze(1)+rotate_half(q)*sin.unsqueeze(1))[0,:,0].float()
   keys=torch.cat((c.sram_k[:,fast],c.stt_k[:,live]),dim=1).float().repeat_interleave(2,dim=0)
   weights=(q[:,None,:]@keys.transpose(1,2)*m.scaling).softmax(-1)
   ranks=weights[:,0,len(fast):].mean(0)
   chosen=sorted(live[torch.topk(ranks,n).indices].tolist());positions=[c.stt_pos[j] for j in chosen]
   assert positions==p._vis_stt_pos[i]
   layer=kw['past_key_values'].layers[i]
   assert layer.keys.shape[2]==len(fast)+n,'Hook ran after append or changed length'
   assert torch.equal(layer.keys[0],torch.cat((c.sram_k[:,fast],c.stt_k[:,chosen]),dim=1))
   assert torch.equal(layer.values[0],torch.cat((c.sram_v[:,fast],c.stt_v[:,chosen]),dim=1))
   stale=p._vis_stt_idx[i]!=chosen
   observations.append(dict(layer=i,chosen=chosen,positions=positions,stale_indices=stale))
  return hook
 for i,l in enumerate(model.model.layers):handles.append(l.self_attn.register_forward_pre_hook(validate(i),with_kwargs=True))
 try:
  with torch.inference_mode():
   ids=torch.arange(24).reshape(1,-1)%64
   out=model(ids,use_cache=True,position_ids=torch.arange(24).reshape(1,-1));raw=out.past_key_values
   apply_tuple_kv_inplace(raw,p(to_tuple_kv(raw)));assert not p.prompt_q_by_layer
   for pos in range(24,32):
    out=model(torch.tensor([[pos%64]]),past_key_values=raw,use_cache=True,position_ids=torch.tensor([[pos]]),output_attentions=True)
    assert torch.isfinite(out.logits).all();past=to_tuple_kv(out.past_key_values)
    expected={}
    for i,c in p.layer_caches.items():
     n=len(p._vis_sram_keep[i]);vals=out.attentions[i][0,:,0].float().mean(0)[n:n+len(p._vis_stt_pos[i])]
     expected[i]=dict(zip(p._vis_stt_pos[i],vals.tolist()))
    p.update_scores(past,out.attentions)
    assert p._stt_attn==expected
    for i,c in p.layer_caches.items():
     if observations[-2+i]['stale_indices']:assert i not in p._stt_attn_idx
     for token,score in expected[i].items():
      if score>0:assert c.stt_last_access[c.stt_pos.index(token)].item()==c._t
    apply_tuple_kv_inplace(raw,p(past))
 finally:
  for h in handles:h.remove()
  selector.close();capture.close();p.reset()
 assert original==[(len(l.self_attn._forward_pre_hooks),l.self_attn.num_key_value_groups) for l in model.model.layers]
 return dict(dtype=str(dtype),decode_steps=8,layer_observations=len(observations),stale_index_fallbacks=sum(o['stale_indices'] for o in observations),expanded_head_rank_matches=True,preappend_payload_matches=True,feedback_positions_and_lru_match=True,hooks_restored=True)

output=Path(__file__).with_name('selector_checks.json')
assert not output.exists(),'Preserve prior evidence'
report={'device':'cpu','checks':[run(torch.float32),run(torch.bfloat16)],'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['experiments/evaluate_query_selection.py','experiments/longbench_eval.py','src/attention.py']}}
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
