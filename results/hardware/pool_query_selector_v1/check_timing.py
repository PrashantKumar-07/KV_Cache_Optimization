"""Preregistered quiet GPU0 current-query versus ordinaryLRU3ABBA guard."""
import gc,hashlib,json,os,subprocess,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from bulk_promotion_policy import BulkPromotionPolicy
from evaluate_query_selection import PromptQueryCapture
from pool_query_selector import PoolQuerySelector
from latency_profile import profile_one
from provenance import atomic_json,runtime_identity


def quiet():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    gpu=next(line for line in lines if line.split(',')[0].strip()=='0');uuid=gpu.split(',')[1].strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    own=[line for line in apps if line.split(',')[0].strip()==uuid]
    other=[line for line in own if int(line.split(',')[1].strip())!=os.getpid()]
    return dict(gpu=gpu,compute_processes=own,quiet=not other,unexpected_compute_processes=other)


def main():
    base=Path(__file__).resolve().parent;output=base/'timing.json';assert not output.exists()
    plan=json.loads((base/'timing_plan.json').read_text());settings=plan['settings']
    report=dict(passed=False,rows=[],quiet_observations=[],runtime=runtime_identity(),pid=os.getpid(),
        plan_sha256=hashlib.sha256((base/'timing_plan.json').read_bytes()).hexdigest())
    try:
        initial=quiet();report['quiet_observations'].append(dict(stage='before_model_load',**initial));assert initial['quiet']
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        prompt=Path(plan['prompt']).read_text();assert hashlib.sha256(Path(plan['prompt']).read_bytes()).hexdigest()==plan['prompt_sha256']
        teachers=Path(plan['teachers']);assert hashlib.sha256(teachers.read_bytes()).hexdigest()==plan['teachers_sha256']
        stream=json.loads(teachers.read_text())['config']['teacher_token_ids']
        for block in range(3):
            for arm in ('control','query','query','control'):
                observation=quiet();report['quiet_observations'].append(dict(stage='before_row',block=block,arm=arm,**observation))
                assert observation['quiet'],'Unexpected GPU0 compute process'
                before=[len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers]
                policy=BulkPromotionPolicy(sram_budget=991,stt_budget=1090,stt_expose_quota=32,start_size=4,recent_size=128,
                    kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
                policy.prompt_query_score_mode='chunked';policy.prompt_query_chunk=128
                capture=PromptQueryCapture(model,policy)
                selector=PoolQuerySelector(model,policy) if arm=='query' else None
                try:
                    result=profile_one(model,tokenizer,policy,16384,n_decode=50,n_warmup=1,n_repeat=1,
                        device='cuda',prompt_text=prompt,forced_token_ids=stream,prefill_last_logits=True)
                    result.update(host_read_bytes=sum(c.host_read_bytes for c in policy.layer_caches.values()),
                        host_write_bytes=sum(c.host_write_bytes for c in policy.layer_caches.values()),
                        native_h2d_calls=policy.manager.h2d_calls,native_d2h_calls=policy.manager.d2h_calls,
                        selected_layers=selector.selected_layers if selector else 0)
                    if selector:assert selector.selected_layers==3200
                finally:
                    if selector:selector.close()
                    capture.close();policy.reset();policy=capture=selector=None
                    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
                assert before==[len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers]
                observation=quiet();report['quiet_observations'].append(dict(stage='after_row',block=block,arm=arm,**observation))
                assert observation['quiet'],'Unexpected GPU0 compute process'
                report['rows'].append(dict(block=block,arm=arm,own_hooks_removed=True,**result));atomic_json(output,report)
                print(block,arm,result['itl_ms'],result['ttft_ms'],flush=True)
        assert len(report['rows'])==12
        report.update(passed=True,limits='Before/after row GPU0 compute-process observations, not continuous utilization monitoring. '
            'Same actualQ/lastlogits ordinaryBulk profiler for both arms. Query warmup selection counters include both trials; '
            'transfer bytes describe final measured trial. No independent quality/faithful-baseline joint win/physicalSTT/adoption claim.')
    except BaseException as e:report['error']=f'{type(e).__name__}: {e}';raise
    finally:atomic_json(output,report)
if __name__=='__main__':main()
