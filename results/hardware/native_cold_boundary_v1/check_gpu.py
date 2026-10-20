"""Frozen cold-boundary bit/selection/lifetime parity and charged stage gate."""
import hashlib,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path[:0]=[str(R/'src'),str(R/'experiments')]
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from bulk_payload_policy import BulkPayloadPolicy
from cold_boundary import visible,kernel
from evaluate_query_selection import PromptQueryCapture
from shared_decode_transition import SharedAppendLease
from latency_profile import profile_one
from compare_cpu_batched import digest
from provenance import atomic_json,runtime_identity
B=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def witness(p,past):
 torch.cuda.synchronize();h=hashlib.sha256()
 for pair in past:
  for t in pair:h.update(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return dict(payload_sha256=h.hexdigest(),selected=p._vis_stt_idx,positions=p._vis_stt_pos,kept=p._vis_sram_keep,lengths=p._vis_len)
def counters(p):return dict(calls=p.manager.h2d_calls,bytes=sum(c.host_read_bytes for c in p.layer_caches.values()),d2h=p.manager.d2h_calls,write_bytes=sum(c.host_write_bytes for c in p.layer_caches.values()))
def prove(p,past,name,report):
 owners={k:v.copy() for k,v in p.cpu_arena.owners.items()};before=digest(p.layer_caches);counts=counters(p);a=p._visible_tuple(past);ref=witness(p,a);mid=counters(p);c=visible(p,past);actual=witness(p,c);after=counters(p)
 assert ref==actual and digest(p.layer_caches)==before and all(np.array_equal(v,p.cpu_arena.owners[k]) for k,v in owners.items())
 assert mid['calls']-counts['calls']==after['calls']-mid['calls']==1 and mid['bytes']-counts['bytes']==after['bytes']-mid['bytes']
 assert after['d2h']==counts['d2h'] and after['write_bytes']==counts['write_bytes']
 report['checks'].append(dict(case=name,exact=True,visible_sha256=ref['payload_sha256'],native_h2d_bytes_per_call=after['bytes']-mid['bytes']))
 return c
@torch.inference_mode()
def main():
 out=B/'gpu_stage.json';assert not out.exists();plan=json.loads((B/'plan.json').read_text());torch.manual_seed(0)
 report=dict(passed=False,checks=[],rows=[],rejections=[],plan_sha256=sha(B/'plan.json'),runtime=runtime_identity());module=kernel();report['binary_path']=module.__file__;report['binary_sha256']=sha(Path(module.__file__))
 model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval();tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
 for hot,cold,reference in [(649,1982,R/plan['control']),(991,1090,R/'results/hardware/cpu_phase_score_guard_v1/resource.json')]:
  ref=json.loads(reference.read_text());p=BulkPayloadPolicy(sram_budget=hot,stt_budget=cold,stt_expose_quota=32,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75);p.prompt_query_score_mode='chunked';p.prompt_query_chunk=128
  hooks=[len(l.self_attn._forward_pre_hooks) for l in model.model.layers];capture=PromptQueryCapture(model,p);lease=SharedAppendLease(model,p);manager=None
  try:
   prompt=R/'results/remediation_v3/latency_prompt.txt';teachers=R/'results/shadow_ablation_full/fixed_stream_20260929.json';stream=json.loads(teachers.read_text())['config']['teacher_token_ids']
   result=profile_one(model,tokenizer,p,16384,n_decode=50,n_warmup=0,n_repeat=1,device='cuda',prompt_text=prompt.read_text(),forced_token_ids=stream,prefill_last_logits=True)
   assert digest(p.layer_caches)==ref['state_sha256'] and p._live_totals()==ref['counters'] and result['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
   for k in ['host_read_bytes','host_write_bytes']:assert sum(getattr(c,k) for c in p.layer_caches.values())==ref['prototype_counts'][k]
   assert p.manager.h2d_calls==ref['prototype_counts']['native_h2d_calls'] and p.manager.d2h_calls==ref['prototype_counts']['native_d2h_calls']
   manager=p.manager;past=tuple(tuple(t.unsqueeze(0).as_strided((1,t.shape[0],hot+32,t.shape[2]),t.unsqueeze(0).stride()) for t in [c.sram_k,c.sram_v]) for c in p.layer_caches.values())
   cases=['actual_final'] if hot==991 else ['actual_final','ragged_tied_nan','small_exposure']
   for case in cases:
    if case!='actual_final':
     for i,n in enumerate(p.cpu_arena.owners['cold_n']):
      flags=p.cpu_arena.owners['shadow'][i,:int(n)]
      if case=='ragged_tied_nan':flags[:]=np.arange(int(n))%(3+i%3)==0
      else:flags[:]=True;flags[:12+i%4]=False
     p.cpu_arena.owners['version']+=1
     values=torch.arange(p.state_arena.cold.numel(),device='cuda').reshape_as(p.state_arena.cold)%7;p.state_arena.cold.copy_(values.to(torch.float32))
     if case=='ragged_tied_nan':p.state_arena.cold[:,1::97]=float('nan')
    proven=prove(p,past,f'{hot}/{cold}/{case}',report)
    if case=='small_exposure':continue
    counts=counters(p);current_calls=candidate_calls=0;state=digest(p.layer_caches)
    for block in range(3):
     for arm in ['current','native','native','current']:
      values=[]
      for step in range(24):
       torch.cuda.synchronize();start=time.perf_counter()
       if arm=='current':output=p._visible_tuple(past);current_calls+=1
       else:output=visible(p,past);candidate_calls+=1
       torch.cuda.synchronize();values.append((time.perf_counter()-start)*1000)
      report['rows'].append(dict(hot=hot,cold=cold,case=case,block=block,arm=arm,step_ms=values[4:],warmup_ms=values[:4]));atomic_json(out,report)
    assert current_calls==candidate_calls==144 and digest(p.layer_caches)==state
    after=counters(p);assert after['calls']-counts['calls']==288 and after['bytes']-counts['bytes']==288*4*2**20
    assert after['d2h']==counts['d2h'] and after['write_bytes']==counts['write_bytes']
    assert witness(p,output)==witness(p,proven)
   if hot==649:
    # Full-shadow/late malformed roots/invalid physicalslot reject before IO.
    for mode in ['empty','late_alias','slot']:
     saved_shadow=p.cpu_arena.owners['shadow'].copy();saved_slot=p.cpu_arena.owners['slots'].copy();cache=p.layer_caches[31];saved_v=cache.sram_v;raw=list(past)
     if mode=='empty':p.cpu_arena.owners['shadow'][:]=True
     elif mode=='late_alias':cache.sram_v=cache.sram_k;raw[31]=(past[31][0],past[31][0])
     else:p.cpu_arena.owners['slots'][31,0]=p.cpu_arena.owners['slots'].shape[1]
     before=witness(p,past);io=counters(p)
     try:
      try:visible(p,tuple(raw))
      except ValueError:pass
      else:raise AssertionError('Malformed batch accepted')
      assert counters(p)==io and witness(p,past)==before
      report['rejections'].append(dict(case=mode,atomic=True))
     finally:cache.sram_v=saved_v;p.cpu_arena.owners['shadow'][:]=saved_shadow;p.cpu_arena.owners['slots'][:]=saved_slot
    reference_visible=p._visible_tuple(past);expected=witness(p,reference_visible);producer=torch.cuda.Event();producer.record();consumer=torch.cuda.Stream();done=torch.cuda.Event()
    with torch.cuda.stream(consumer):
     consumer.wait_event(producer);value=visible(p,past);done.record()
    pressure=torch.empty(4*1024*1024,dtype=torch.bfloat16,device='cuda');pressure.fill_(1);done.synchronize();assert witness(p,value)==expected
    report['nondefault_stream_handoff_exact']=True;del pressure,value
  finally:
   lease.close();capture.close();p.reset();torch.cuda.synchronize()
   assert hooks==[len(l.self_attn._forward_pre_hooks) for l in model.model.layers] and not lease.handles and not lease.pending and p.manager is None
   if manager:assert manager.pool is None and manager.bulk_pending is None
   atomic_json(out,report)
 summary=[]
 for hot,cold,case in [(649,1982,'actual_final'),(649,1982,'ragged_tied_nan'),(991,1090,'actual_final')]:
  rows=[r for r in report['rows'] if (r['hot'],r['cold'],r['case'])==(hot,cold,case)];data={a:[v for r in rows if r['arm']==a for v in r['step_ms']] for a in ['current','native']}
  med={a:float(np.median(v)) for a,v in data.items()};p95={a:float(np.quantile(v,.95)) for a,v in data.items()};saving=med['current']-med['native'];guards=dict(saved_ms=saving>=4,median=med['native']/med['current']<=.95,p95=p95['native']/p95['current']<=1.02)
  summary.append(dict(hot=hot,cold=cold,case=case,median_ms=med,p95_ms=p95,saved_ms=saving,guards=guards,passed=all(guards.values())))
 report.update(passed=True,summary=summary,actual_stage_gate_passed=all(r['passed'] for r in summary),all_owned_hooks_leases_pool_released=True,limits=plan['limits'],full_goal_complete=False);atomic_json(out,report);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
