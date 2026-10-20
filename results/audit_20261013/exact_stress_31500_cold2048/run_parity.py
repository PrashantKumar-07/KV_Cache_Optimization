"""Current full-model GPU-policy/native-controller logits and full KV parity."""
from contextlib import ExitStack
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from longbench_eval import TieredKVPolicy
from exact_cold_boundary_policy import ExactColdBoundaryPolicy
from exact_pinned_memory import stats as allocation_stats
from evaluate_query_selection import PromptQueryCapture
from shared_decode_transition import SharedAppendLease
from latency_profile import profile_one
from provenance import atomic_json,model_artifact_identity,runtime_identity
plan=json.loads((BASE/'plan.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (BASE/'parity.json').exists()
assert all(sha(ROOT/name)==value for name,value in plan['source_sha256'].items())
stream=json.loads((ROOT/plan['teacher_stream']).read_text())['config']['teacher_token_ids']
assert sha(ROOT/plan['teacher_stream'])==plan['teacher_stream_sha256']
assert sha(ROOT/plan['prompt'])==plan['prompt_sha256']
report={'passed':False,'arms':{},'plan_sha256':sha(BASE/'plan.json'),
    'model_identity':model_artifact_identity('../models/mistral-7b-instruct'),'runtime':runtime_identity()}

def tensor_hash(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

class Record:
    def __init__(self,**options):
        super().__init__(**options);self.trace=[]
    def snapshot(self,visible):
        torch.cuda.synchronize()
        layers={}
        for index,cache in self.layer_caches.items():
            layers[str(index)]={'hot_pos':list(cache.sram_pos),'cold_pos':list(cache.stt_pos),
                'cold_shadow':cache.stt_shadow.tolist(),
                'hot_k':tensor_hash(cache.sram_k),'hot_v':tensor_hash(cache.sram_v),
                'cold_k':tensor_hash(cache.stt_k),'cold_v':tensor_hash(cache.stt_v),
                'visible_k':tensor_hash(visible[index][0]),'visible_v':tensor_hash(visible[index][1])}
        self.trace.append(layers)
    def _bifurcate(self,*args,**options):
        out=super()._bifurcate(*args,**options);self.snapshot(out);return out
    def _decode_step(self,*args,**options):
        out=super()._decode_step(*args,**options);self.snapshot(out);return out
class Control(Record,TieredKVPolicy):pass
class Candidate(Record,ExactColdBoundaryPolicy):pass

try:
    torch.manual_seed(0)
    model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,
        device_map='cuda',attn_implementation='eager').eval()
    assert model.config.sliding_window is None
    tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
    records={}
    for arm,kind in [('control',Control),('candidate',Candidate)]:
        hooks_before=(len(model._forward_pre_hooks),len(model._forward_hooks),
            [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers])
        policy=kind(sram_budget=1024,stt_budget=2048,stt_expose_quota=32,start_size=4,
            recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
        logits=[]
        with ExitStack() as cleanup:
            cleanup.callback(policy.reset)
            capture=PromptQueryCapture(model,policy);cleanup.callback(capture.close)
            lease=SharedAppendLease(model,policy) if arm=='candidate' else None
            if lease:cleanup.callback(lease.close)
            handle=model.register_forward_hook(lambda m,args,out:logits.append(out.logits.detach().float().cpu().numpy()))
            cleanup.callback(handle.remove)
            result=profile_one(model,tokenizer,policy,31500,n_decode=50,n_warmup=0,n_repeat=1,
                device='cuda',prompt_text=(ROOT/plan['prompt']).read_text(),forced_token_ids=stream,prefill_last_logits=True)
            np.savez_compressed(BASE/f'{arm}_logits.npz',**{str(i):v for i,v in enumerate(logits)})
            row={'trace':policy.trace,'predicted_token_ids':result['trials'][0]['predicted_token_ids'],
                'forwards':len(logits),'cold_steps':getattr(policy,'cold_boundary_steps',None),
                'cpu_append_calls':getattr(policy,'cpu_append_calls',None),
                'cpu_fallback_steps':getattr(policy,'cpu_fallback_steps',None)}
            records[arm]=row
        torch.cuda.synchronize();torch.cuda.empty_cache()
        row['allocation_stats_after_reset']=allocation_stats()
        assert allocation_stats()['live_allocations']==0
        row['hooks_closed']=hooks_before==(len(model._forward_pre_hooks),len(model._forward_hooks),
            [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers])
        report['arms'][arm]=row
        atomic_json(BASE/'parity.json',report)
        print(arm,'forwards',len(logits),'trace',len(row['trace']),flush=True)
    left,right=records['control'],records['candidate']
    report['trace_matches']=left['trace']==right['trace']
    report['prediction_matches']=left['predicted_token_ids']==right['predicted_token_ids']
    a,b=[np.load(BASE/f'{arm}_logits.npz') for arm in ('control','candidate')]
    report['full_logits_bitwise_match']=all(np.array_equal(a[str(i)],b[str(i)]) for i in range(51))
    report['max_logit_abs_difference']=max(float(np.max(np.abs(a[str(i)]-b[str(i)]))) for i in range(51))
    report['passed']=bool(report['trace_matches'] and report['prediction_matches'] and report['full_logits_bitwise_match']
        and all(row['hooks_closed'] and row['forwards']==51 and len(row['trace'])==51 for row in records.values())
        and right['cold_steps']==50 and right['cpu_append_calls']==50 and right['cpu_fallback_steps']==0)
    assert report['passed'],'Registered strict full-model parity gate failed; preserve result for diagnosis'
finally:
    atomic_json(BASE/'parity.json',report)
