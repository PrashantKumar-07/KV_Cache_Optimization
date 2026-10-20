"""Existing policy-only profiler on current ordinary bulk shared path."""
import hashlib,json,pstats,sys,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from profile_cpu_batched import ProfileCalls
from bulk_promotion_policy import BulkPromotionPolicy
from compare_cpu_batched import digest
from evaluate_query_selection import PromptQueryCapture
from latency_profile import profile_one
from provenance import atomic_json,runtime_identity
import importlib.util
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('quiet_check',ROOT/'results/hardware/pool_query_selector_v1/check_timing.py')
quiet_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(quiet_module)
class ProfileBulk(ProfileCalls,BulkPromotionPolicy):pass
def main():
    output=BASE/'profile.json';assert not output.exists()
    plan=json.loads((BASE/'profile_plan.json').read_text());refpath=ROOT/plan['reference']
    assert hashlib.sha256(refpath.read_bytes()).hexdigest()==plan['reference_sha256'];ref=json.loads(refpath.read_text())
    report=dict(passed=False,runtime=runtime_identity(),plan_sha256=hashlib.sha256((BASE/'profile_plan.json').read_bytes()).hexdigest(),quiet_observations=[])
    policy=capture=None
    try:
        obs=quiet_module.quiet();assert obs['quiet'];report['quiet_observations'].append(obs)
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        before=[len(l.self_attn._forward_pre_hooks) for l in model.model.layers]
        policy=ProfileBulk(sram_budget=991,stt_budget=1090,stt_expose_quota=32,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
        policy.geometry=[];policy.trace_cuda=False
        policy.prompt_query_score_mode='chunked';policy.prompt_query_chunk=128;capture=PromptQueryCapture(model,policy)
        teachers=json.loads((ROOT/'results/shadow_ablation_full/fixed_stream_20260929.json').read_text())['config']['teacher_token_ids']
        timing=profile_one(model,tokenizer,policy,16384,n_decode=50,n_warmup=0,n_repeat=1,device='cuda',prompt_text=(ROOT/'results/remediation_v3/latency_prompt.txt').read_text(),forced_token_ids=teachers,prefill_last_logits=True)
        state=digest(policy.layer_caches);counters=policy._live_totals()
        assert state==ref['state_sha256'] and counters==ref['counters']
        assert timing['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
        payload={f:sum(getattr(c,f) for c in policy.layer_caches.values()) for f in ['host_read_bytes','host_write_bytes']}
        assert all(v==ref['prototype_counts'][f] for f,v in payload.items())
        assert policy.manager.h2d_calls==ref['native_h2d_calls'] and policy.manager.d2h_calls==ref['native_d2h_calls']
        policy.profiler.dump_stats(str(BASE/'policy.prof'))
        text=io.StringIO();pstats.Stats(policy.profiler,stream=text).strip_dirs().sort_stats('cumulative').print_stats(65)
        (BASE/'policy.prof.txt').write_text(text.getvalue())
        report.update(frozen50predictions_state_counters_payload_calls_replayed=True,state_sha256=state,counters=counters,payload=payload,profiled_timing_not_latency_gate=timing,geometry=policy.geometry)
        capture.close();policy.reset()
        assert before==[len(l.self_attn._forward_pre_hooks) for l in model.model.layers]
        obs=quiet_module.quiet();assert obs['quiet'];report['quiet_observations'].append(obs)
        report.update(passed=True,hooks_restored=True,owned_host_after_cleanup_bytes=policy.manager.owned_host_allocator_block_bytes if policy.manager else 0,limits=plan['limits'])
    except BaseException as error:report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        if capture:capture.close()
        if policy:policy.reset()
        atomic_json(output,report)
if __name__=='__main__':main()
