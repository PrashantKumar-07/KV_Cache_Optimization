"""Reuse the frozen envelope profiler with scoped research leases."""
import gc,hashlib,importlib.util,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from early_shared_cpu_phase import EarlySharedPolicy,SharedPrefillLease
from shared_decode_transition import SharedAppendLease
from compare_cpu_batched import digest
from provenance import atomic_json,runtime_identity
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    output=BASE/'cost.json';assert not output.exists()
    plan=json.loads((BASE/'cost_plan.json').read_text());refs={}
    for arm in ['cpu','snap']:
        path=ROOT/plan[arm];assert sha(path)==plan[arm+'_sha256'];refs[arm]=json.loads(path.read_text())
    spec=importlib.util.spec_from_file_location('quiet_observer',ROOT/'results/hardware/pool_query_selector_v1/check_timing.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
    source=(ROOT/'experiments/compare_snap_tier_envelope.py').read_text()
    source=source.replace('policy = capture = past = None','policy = capture = past = None;leases=[]')
    source=source.replace('capacity=974','capacity=682')
    source=source.replace("cfg = SETTINGS[arm]","cfg = dict(hot=649,cold=1982,exposed=32)")
    source=source.replace('policy_cls = FusedBoundedPoolPolicy if cpu_fused_bounded else DirectScorePolicy','policy_cls = EarlySharedPolicy')
    source=source.replace('capture = PromptQueryCapture(model, policy)','capture = PromptQueryCapture(model, policy)\n                leases=[SharedPrefillLease(model,policy),SharedAppendLease(model,policy)]')
    source=source.replace("position_ids=torch.arange(ids.shape[1], device='cuda')[None]), 'cuda')","position_ids=torch.arange(ids.shape[1], device='cuda')[None],logits_to_keep=1), 'cuda')")
    source=source.replace('if policy is not None: policy.reset()','for lease in leases:lease.close()\n            if policy is not None: policy.reset()')
    source=source.replace("counters=policy._live_totals() if policy is not None else None,", "state_sha256=digest(policy.layer_caches) if policy else cache_hash(past),\n                    native_h2d_calls=policy.manager.h2d_calls if policy else 0,native_d2h_calls=policy.manager.d2h_calls if policy else 0,\n                    early_shared_layers=policy.shared_prefill_layers if policy else 0,\n                    counters=policy._live_totals() if policy is not None else None,")
    scope=dict(__name__='scoped_envelope_profiler',__file__=str(ROOT/'experiments/compare_snap_tier_envelope.py'))
    exec(compile(source,'<scoped-envelope-profiler>','exec'),scope)
    snap_spec=importlib.util.spec_from_file_location('snap_hash_reference',ROOT/'results/hardware/prompt_boundary_comparison_v1/audit_snap_hashed.py');snap_module=importlib.util.module_from_spec(snap_spec);snap_spec.loader.exec_module(snap_module)
    scope.update(EarlySharedPolicy=EarlySharedPolicy,SharedPrefillLease=SharedPrefillLease,SharedAppendLease=SharedAppendLease,digest=digest,cache_hash=snap_module.cache_hash)
    report=dict(passed=False,rows=[],quiet_observations=[],runtime=runtime_identity(),plan_sha256=sha(BASE/'cost_plan.json'))
    try:
        q=observer.quiet();report['quiet_observations'].append(q);assert q['quiet']
        torch.manual_seed(0);model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval();tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        prompt=ROOT/plan['prompt'];teachers=ROOT/plan['teachers'];assert sha(prompt)==plan['prompt_sha256'] and sha(teachers)==plan['teachers_sha256']
        ids=tokenizer(prompt.read_text(),return_tensors='pt').input_ids[:,:16384].to('cuda');stream=torch.tensor([json.loads(teachers.read_text())['config']['teacher_token_ids'][:50]],device='cuda')
        for i,arm in enumerate(plan['order']):
            q=observer.quiet();report['quiet_observations'].append(q);assert q['quiet']
            result=scope['measure'](model,ids,stream,'snap' if arm=='snap' else 'cpu_direct')
            ref=refs[arm];assert result['predicted_token_ids']==ref['predicted_token_ids']
            if arm=='cpu':
                assert result['state_sha256']==ref['state_sha256'] and result['counters']==ref['counters']
                assert result['early_shared_layers']==32
                for f,key in [('h2d_payload_bytes','host_read_bytes'),('d2h_payload_bytes','host_write_bytes'),('native_h2d_calls','native_h2d_calls'),('native_d2h_calls','native_d2h_calls')]:assert result[f]==ref['prototype_counts'][key]
            else:assert result['state_sha256']==ref['final_cache_sha256']
            q=observer.quiet();report['quiet_observations'].append(q);assert q['quiet']
            report['rows'].append(dict(block=i//4,arm=arm,**result));atomic_json(output,report);print(i,arm,statistics.median(result['step_ms']),result['ttft_ms'],flush=True)
        report['passed']=True
    except BaseException as e:report['error']=f'{type(e).__name__}: {e}';raise
    finally:atomic_json(output,report)
if __name__=='__main__':main()
