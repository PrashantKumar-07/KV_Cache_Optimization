"""Stationary full cold-boundary optimistic screen, no new policy code."""
import hashlib,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path[:0]=[str(R/'src'),str(R/'experiments')]
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from bulk_payload_policy import BulkPayloadPolicy
from evaluate_query_selection import PromptQueryCapture
from shared_decode_transition import SharedAppendLease
from latency_profile import profile_one
from compare_cpu_batched import digest
from provenance import atomic_json
B=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def visible_hash(past):
 h=hashlib.sha256()
 for pair in past:
  for t in pair:h.update(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
@torch.inference_mode()
def main():
 plan=json.loads((B/'screen_plan.json').read_text());refp=R/plan['reference'];assert sha(refp)==plan['reference_sha256'];ref=json.loads(refp.read_text())
 torch.manual_seed(0);model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval();tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
 p=BulkPayloadPolicy(sram_budget=649,stt_budget=1982,stt_expose_quota=32,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75);p.prompt_query_score_mode='chunked';p.prompt_query_chunk=128
 hooks=[len(l.self_attn._forward_pre_hooks) for l in model.model.layers];capture=PromptQueryCapture(model,p);lease=SharedAppendLease(model,p);manager=None;report=dict(passed=False,rows=[],plan_sha256=sha(B/'screen_plan.json'))
 try:
  prompt=R/'results/remediation_v3/latency_prompt.txt';teachers=R/'results/shadow_ablation_full/fixed_stream_20260929.json'
  assert sha(prompt)==ref['prompt_sha256'] and sha(teachers)==ref['teacher_source_sha256']
  stream=json.loads(teachers.read_text())['config']['teacher_token_ids'];result=profile_one(model,tokenizer,p,16384,n_decode=50,n_warmup=0,n_repeat=1,device='cuda',prompt_text=prompt.read_text(),forced_token_ids=stream,prefill_last_logits=True)
  manager=p.manager;state=digest(p.layer_caches);assert state==ref['state_sha256'] and p._live_totals()==ref['counters'] and result['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
  for k in ['host_read_bytes','host_write_bytes']:assert sum(getattr(c,k) for c in p.layer_caches.values())==ref['prototype_counts'][k]
  assert manager.h2d_calls==76 and manager.d2h_calls==57 and p.bulk_payload_steps==50
  past=[]
  for c in p.layer_caches.values():
   pair=[]
   for t in [c.sram_k,c.sram_v]:
    raw=t.unsqueeze(0);pair.append(raw.as_strided((1,t.shape[0],t.shape[1]+32,t.shape[2]),raw.stride()))
   past.append(tuple(pair))
  past=tuple(past);initial_visible=visible_hash(past);owners={k:v.copy() for k,v in p.cpu_arena.owners.items()};calls=0;initial_native=manager.h2d_calls;initial_bytes=sum(c.host_read_bytes for c in p.layer_caches.values())
  for block in range(3):
   for arm in ['current','zero_work','zero_work','current']:
    values=[]
    for step in range(24):
     torch.cuda.synchronize();start=time.perf_counter()
     if arm=='current':output=p._visible_tuple(past);calls+=1
     torch.cuda.synchronize();values.append((time.perf_counter()-start)*1000)
    report['rows'].append(dict(block=block,arm=arm,step_ms=values[4:],warmup_ms=values[:4]));atomic_json(B/'screen.json',report)
  assert digest(p.layer_caches)==state and visible_hash(past)==initial_visible
  assert all(np.array_equal(v,p.cpu_arena.owners[k]) for k,v in owners.items())
  assert calls==144 and manager.h2d_calls-initial_native==calls
  extra=sum(c.host_read_bytes for c in p.layer_caches.values())-initial_bytes;assert extra==calls*4*2**20
  med={arm:float(np.median([v for row in report['rows'] if row['arm']==arm for v in row['step_ms']])) for arm in ['current','zero_work']};saving=med['current']-med['zero_work']
  report.update(passed=True,model_frozen_replay_exact=True,cache_visible_and_cpuowners_unchanged=True,charged_h2d_calls=calls,charged_h2d_bytes=extra,measured_calls_per_arm=120,median_ms=med,optimistic_saving_ms=saving,can_meet_4ms_screen=saving>=4,limits=plan['limits'],full_goal_complete=False)
 finally:
  lease.close();capture.close();p.reset();torch.cuda.synchronize()
  assert hooks==[len(l.self_attn._forward_pre_hooks) for l in model.model.layers] and not lease.handles and not lease.pending and p.manager is None
  if manager:assert manager.pool is None and manager.bulk_pending is None
  report['owned_hooks_leases_pool_released']=True;atomic_json(B/'screen.json',report)
 print(json.dumps({k:report[k] for k in ['median_ms','optimistic_saving_ms','can_meet_4ms_screen']}))
if __name__=='__main__':main()
