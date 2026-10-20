# Exact command submitted for the already-observed tiny GPU1 check.
# This artifact was saved later; do not run while other GPU jobs are active.
CUDA_VISIBLE_DEVICES=1 ../venv_kvcache/bin/python - <<'PY'
import sys,json,torch
sys.path[:0]=['experiments','src','tests']
from transformers import MistralConfig,MistralForCausalLM
from test_october_audit import FixedTokenizer
from longbench_eval import TieredKVPolicy,generate_with_policy
from direct_shared_pool import DirectScorePolicy
assert torch.cuda.device_count()==1
torch.cuda.set_device(0);torch.manual_seed(127)
cfg=MistralConfig(num_hidden_layers=2,hidden_size=512,intermediate_size=768,num_attention_heads=4,num_key_value_heads=2,vocab_size=128,max_position_embeddings=4096,sliding_window=None,attn_implementation='eager')
m=MistralForCausalLM(cfg).to(device='cuda',dtype=torch.bfloat16).eval()
checks=[]
for hot,cold in [(8,16),(32,32)]:
 policies=[cls(sram_budget=hot,stt_budget=cold,start_size=1,recent_size=2,prefill_chunk_size=4,kv_dtype=torch.bfloat16) for cls in (TieredKVPolicy,DirectScorePolicy)]
 results=[];states=[]
 try:
  for p in policies:
   result=generate_with_policy(m,FixedTokenizer(),'prompt',3,p,'cuda')
   torch.cuda.synchronize();results.append(result)
   state={i:(list(c.sram_pos),list(c.stt_pos),c.sram_k.detach().cpu().clone(),c.stt_k.detach().cpu().clone()) for i,c in p.layer_caches.items()}
   states.append(state)
   assert all(c.metrics.prompt_len==20 for c in p.layer_caches.values())
  assert results[0]==results[1],results
  for i in states[0]:
   assert states[0][i][:2]==states[1][i][:2]
   for x,y in zip(states[0][i][2:],states[1][i][2:]):assert torch.equal(x,y)
  checks.append(dict(hot=hot,cold=cold,prediction=results[0],parity=True))
 finally:
  for p in policies:p.reset();p.reset()
  assert policies[1].manager is None
print(json.dumps(dict(device='CUDA_VISIBLE_DEVICES=1',passed=True,checks=checks,peak_allocated_bytes=torch.cuda.max_memory_allocated()),indent=2))
PY
