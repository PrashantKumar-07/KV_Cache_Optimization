"""Real multi-query score shapes must preserve ordinary fallback semantics."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import pytest
import torch
from batched_score_policy import batched_update
from longbench_eval import TieredKVPolicy
from probe_batched_score_updates import fixture,assert_equal

@pytest.mark.parametrize('shape', [(1,0,1,43), (1,2,0,43), (0,2,1,43),
                                  (2,2,1,43), (2,1,43)])
@pytest.mark.parametrize('update', [TieredKVPolicy.update_scores, batched_update])
def test_invalid_feedback_rejects_before_any_layer_mutates(shape, update):
 weights=[torch.ones(1,2,1,43) for _ in range(4)]
 weights[-1]=torch.empty(shape)
 initial=torch.ones(4,40)
 policy=fixture(weights,initial,2)
 policy._stt_attn={0:{100:0.5}}; policy._stt_attn_idx={0:{1:0.5}}
 with pytest.raises(ValueError,match='positive heads and queries'):
  update(policy,[(None,None)]*4,weights)
 for layer,cache in policy.layer_caches.items():
  torch.testing.assert_close(cache.sram_cum,initial[layer])
  assert torch.count_nonzero(cache.stt_last_access)==0
 assert policy._stt_attn=={0:{100:0.5}} and policy._stt_attn_idx=={0:{1:0.5}}

def test_empty_prompt_query_heads_reject_before_cache_initializes():
 policy=TieredKVPolicy(sram_budget=4,stt_budget=4,start_size=1,recent_size=2)
 policy.prompt_q_by_layer={0:torch.zeros(0,2,2)}
 keys=torch.zeros(1,2,8,2)
 with pytest.raises(ValueError,match='Captured prompt query shape'):
  policy(((keys,keys),))
 assert not policy.layer_caches

@pytest.mark.parametrize('queries',[1,2,32,64])
def test_multi_token_attention_uses_ordinary_score_fallback(queries):
 generator=torch.Generator().manual_seed(7)
 weights=[torch.rand(1,2,queries,40+2+queries,generator=generator) for _ in range(4)]
 initial=torch.rand(4,40,generator=generator)
 ordinary,batched=(fixture(weights,initial,2) for _ in range(2))
 TieredKVPolicy.update_scores(ordinary,[(None,None)]*4,weights)
 batched_update(batched,[(None,None)]*4,weights)
 assert_equal(ordinary,batched)
 for i in range(4):
  expected=weights[i][0].float().mean(0).mean(0)
  torch.testing.assert_close(batched.layer_caches[i].sram_cum,initial[i]+expected[:40])
  torch.testing.assert_close(torch.tensor(list(batched._stt_attn[i].values())),expected[40:42])
