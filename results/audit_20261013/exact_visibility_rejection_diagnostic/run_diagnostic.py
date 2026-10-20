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
import cold_boundary_policy as cold_policy_module
plan=json.loads((BASE/'plan.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (BASE/'parity.json').exists()
assert all(sha(ROOT/name)==value for name,value in plan['source_sha256'].items())
stream=json.loads((ROOT/plan['teacher_stream']).read_text())['config']['teacher_token_ids']
assert sha(ROOT/plan['teacher_stream'])==plan['teacher_stream_sha256']
assert sha(ROOT/plan['prompt'])==plan['prompt_sha256']
report={'passed':False,'diagnostic_complete':False,'arms':{},'plan_sha256':sha(BASE/'plan.json'),
    'model_identity':model_artifact_identity('../models/mistral-7b-instruct'),'runtime':runtime_identity()}
assert report['model_identity']==plan['model_identity'], 'Local model changed after diagnostic preregistration'

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

def tensor_info(t):
    if t is None:return None
    return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),
        data_ptr=t.data_ptr(),storage_ptr=t.untyped_storage().data_ptr(),storage_bytes=t.untyped_storage().nbytes(),
        storage_offset=t.storage_offset(),requires_grad=t.requires_grad)

def boundary_context(policy,past):
    cpu=policy.cpu_arena;arena=policy.state_arena;manager=policy.manager
    result=dict(decode_step_count=policy.decode_step_count,cold_steps=policy.cold_boundary_steps,
        cpu_append_calls=policy.cpu_append_calls,cpu_fallback_steps=policy.cpu_fallback_steps,
        bulk_payload_steps=policy.bulk_payload_steps,bulk_payload_fallback_steps=policy.bulk_payload_fallback_steps,
        cpu_present=cpu is not None,state_present=arena is not None,manager_present=manager is not None,
        pool=None if manager is None else tensor_info(manager.pool),layers={})
    if cpu is not None:result['canonical_counts']={k:cpu.owners[k].tolist() for k in ['hot_n','cold_n','version']}
    if arena is not None:result['cold_metadata']=tensor_info(arena.cold)
    if manager is not None:result['manager_pending']=len(manager.pending);result['manager_bulk_pending']=manager.bulk_pending is not None
    for i,c in policy.layer_caches.items():
        result['layers'][str(i)]=dict(hot_count=len(c.sram_pos),cold_count=len(c.stt_pos),
            live_count=int((~c.stt_shadow).sum()),hot_k=tensor_info(c.sram_k),hot_v=tensor_info(c.sram_v),
            raw_k=tensor_info(past[i][0]),raw_v=tensor_info(past[i][1]),timestamps=tensor_info(c.stt_last_access),
            expected_timestamp_ptr=None if arena is None else arena.cold[i].data_ptr(),
            pending_keep=c.pending_keep is not None,prefetched=c._prefetched is not None,
            cpu_commands=c._cpu_commands is not None,attached_cpu=c._cpu_arena is cpu,
            attached_state=c._state_arena is arena,owned_manager=c.manager is manager)
    return result

def diagnostic_visible(original,events):
    def invoke(policy,past):
        try:context=boundary_context(policy,past)
        except Exception as error:context={'probe_context_error':repr(error)}
        row={'attempt':len(events),'context_before':context}
        try:
            result=original(policy,past)
        except ValueError as error:
            row.update(native=False,value_error=str(error));events.append(row)
            raise # unchanged policy catches and executes its existing fallback
        row.update(native=True,value_error=None);events.append(row)
        return result
    return invoke

def restore_visible(original):cold_policy_module.visible=original

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
        policy=kind(sram_budget=1024,stt_budget=256,stt_expose_quota=32,start_size=4,
            recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
        logits=[]
        with ExitStack() as cleanup:
            cleanup.callback(policy.reset)
            visibility_events=[]
            if arm=='candidate':
                original_visible=cold_policy_module.visible
                cold_policy_module.visible=diagnostic_visible(original_visible,visibility_events)
                cleanup.callback(restore_visible,original_visible)
            capture=PromptQueryCapture(model,policy);cleanup.callback(capture.close)
            lease=SharedAppendLease(model,policy) if arm=='candidate' else None
            if lease:cleanup.callback(lease.close)
            handle=model.register_forward_hook(lambda m,args,out:logits.append(out.logits.detach().float().cpu().numpy()))
            cleanup.callback(handle.remove)
            result=profile_one(model,tokenizer,policy,8192,n_decode=50,n_warmup=0,n_repeat=1,
                device='cuda',prompt_text=(ROOT/plan['prompt']).read_text(),forced_token_ids=stream,prefill_last_logits=True)
            np.savez_compressed(BASE/f'{arm}_logits.npz',**{str(i):v for i,v in enumerate(logits)})
            row={'trace':policy.trace,'predicted_token_ids':result['trials'][0]['predicted_token_ids'],
                'forwards':len(logits),'cold_steps':getattr(policy,'cold_boundary_steps',None),
                'cpu_append_calls':getattr(policy,'cpu_append_calls',None),
                'cpu_fallback_steps':getattr(policy,'cpu_fallback_steps',None),
                'cold_boundary_fallback_calls':getattr(policy,'cold_boundary_fallback_calls',None),
                'bulk_payload_steps':getattr(policy,'bulk_payload_steps',None),
                'bulk_payload_fallback_steps':getattr(policy,'bulk_payload_fallback_steps',None),
                'visibility_events':visibility_events}
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
    report['registered_native_coverage_passed']=report['passed']
    report['diagnostic_behavioral_parity']=bool(report['trace_matches'] and report['prediction_matches'] and report['full_logits_bitwise_match'])
    report['diagnostic_complete']=bool(report['diagnostic_behavioral_parity'] and right['cpu_append_calls']==50 and right['cpu_fallback_steps']==0
        and all(row['hooks_closed'] and row['allocation_stats_after_reset']['live_allocations']==0 for row in records.values()))
    assert report['diagnostic_complete'],'Diagnostic failed; preserve raw evidence'
    assert len(right['visibility_events'])==51 and sum(row['native'] for row in right['visibility_events'])==right['cold_steps']
    assert right['cold_steps']==41 and right['cold_boundary_fallback_calls']==10,'Prior pathcoverage changed; preserve for source diagnosis, not a favorable retry'
    assert report['registered_native_coverage_passed'] is False
finally:
    atomic_json(BASE/'parity.json',report)
