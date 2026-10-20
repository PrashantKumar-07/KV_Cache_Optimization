"""Fixed-length model-level isolation across interleaved and recycled requests."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from compare_cpu_batched import GpuPolicy, ScorePolicy, DirectPolicy, BoundedPolicy, FusedPolicy, FusedBoundedPolicy, digest
from evaluate_query_selection import PromptQueryCapture
from longbench_eval import to_tuple_kv, apply_tuple_kv_inplace, _attn_impl
from provenance import atomic_json, runtime_identity


def start(model, ids, mode, direct=False, *, bounded=False, fused=False, fused_bounded=False, hot=1024, cold=2048):
    policy = (GpuPolicy if mode == 'gpu' else FusedBoundedPolicy if fused_bounded else FusedPolicy if fused else BoundedPolicy if bounded else DirectPolicy if direct else ScorePolicy)(sram_budget=hot,
        stt_budget=cold,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,
        sttram_bifurcation_frac=.75)
    policy.record_full_state = True
    policy.prompt_query_score_mode = 'chunked';policy.prompt_query_chunk = 128
    capture = PromptQueryCapture(model,policy)
    try:
        with _attn_impl(model,'sdpa'):
            out = model(ids,use_cache=True,position_ids=torch.arange(ids.shape[1],device='cuda')[None])
        past = out.past_key_values
        apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
        token = out.logits[:,-1:].argmax(dim=-1)
        return dict(policy=policy,past=past,token=token,tokens=[token.item()],ctx=ids.shape[1],step=0)
    except BaseException:
        policy.reset();raise
    finally:
        capture.close()


def advance(model, state):
    policy = state['policy']
    position = torch.tensor([[state['ctx']+state['step']]],device='cuda')
    out = model(state['token'],past_key_values=state['past'],use_cache=True,
        position_ids=position,output_attentions=True)
    past = out.past_key_values;tup = to_tuple_kv(past)
    policy.update_scores(tup,out.attentions)
    apply_tuple_kv_inplace(past,policy(tup))
    token = out.logits[:,-1:].argmax(dim=-1)
    state.update(past=past,token=token,step=state['step']+1)
    state['tokens'].append(token.item())


def finish(state):
    policy = state['policy']
    result = dict(predicted_token_ids=state['tokens'],state_sha256=digest(policy.layer_caches),
        every_boundary_sha256=policy.state_history,
        history_sha256=hashlib.sha256(json.dumps(policy.history,sort_keys=True).encode()).hexdigest(),
        counters=policy._live_totals())
    policy.reset()
    assert not policy.layer_caches and getattr(policy,'manager',None) is None
    state.clear();gc.collect()
    return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--direct',action='store_true',help='Direct shared initialization for interleaved arms')
    p.add_argument('--bounded',action='store_true',help='Bounded direct pool for interleaved arms')
    p.add_argument('--fused',action='store_true',help='Experimental fused coordinator/compaction for interleaved arms')
    p.add_argument('--fused-bounded',action='store_true',help='Composed bounded slots and fused update for interleaved arms')
    p.add_argument('--hot',type=int,default=1024)
    p.add_argument('--cold',type=int,default=2048)
    args=p.parse_args()
    if sum((args.direct,args.bounded,args.fused,args.fused_bounded))>1:p.error('Choose one direct pool variant')
    if args.output.exists():raise FileExistsError(f'Preserve evidence: {args.output}')
    prompt_path = Path('results/remediation_v3/latency_prompt.txt')
    source_files = ('experiments/check_cpu_interleaved_model.py','experiments/compare_cpu_batched.py',
        'experiments/evaluate_query_selection.py','experiments/batched_score_policy.py',
        'experiments/cpu_batched_cache.py','experiments/cpu_slot_cache.py','experiments/direct_shared_pool.py','experiments/bounded_pinned_pool.py')
    report = dict(passed=False,direct_interleaved=args.direct,bounded_interleaved=args.bounded,fused_interleaved=args.fused,fused_bounded_interleaved=args.fused_bounded,hot=args.hot,cold=args.cold,runtime=runtime_identity(),rows=[],prompt_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files})
    live = {}
    try:
        if args.fused or args.fused_bounded:
            from fused_hot_compaction import kernel
            report['binary_sha256']=hashlib.sha256(Path(kernel().__file__).read_bytes()).hexdigest()
            report['fused_source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,
                ('experiments/fused_eviction_policy.py','experiments/fused_hot_compaction.py','experiments/fused_bounded_pool.py',
                 'experiments/native_hot_compact.cpp','experiments/native_hot_compact.cu'))}
        torch.manual_seed(0)
        path='../models/mistral-7b-instruct'
        model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(path)
        ids=tokenizer(prompt_path.read_text(),return_tensors='pt').input_ids.to('cuda')
        assert ids.shape[1]>=12288
        requests={'A':(ids[:,:4096],12),'B':(ids[:,:8192],24),'C':(ids[:,8192:12288],12)}
        report['input_sha256']={name:hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest() for name,(tokens,_) in requests.items()}
        controls={}
        for mode in ('gpu','cpu'):
            for name,(tokens,steps) in requests.items():
                live[name]=start(model,tokens,mode,hot=args.hot,cold=args.cold)
                for _ in range(steps):advance(model,live[name])
                result=finish(live.pop(name));controls[(mode,name)]=result
                report['rows'].append(dict(mode=mode,request=name,**result));atomic_json(args.output,report)
                if mode=='cpu':assert result==controls[('gpu',name)],name
                print(mode,name,'isolated passes',flush=True)
                torch.cuda.empty_cache()
        live['A']=start(model,requests['A'][0],'cpu',args.direct,bounded=args.bounded,fused=args.fused,fused_bounded=args.fused_bounded,hot=args.hot,cold=args.cold);live['B']=start(model,requests['B'][0],'cpu',args.direct,bounded=args.bounded,fused=args.fused,fused_bounded=args.fused_bounded,hot=args.hot,cold=args.cold)
        for step in range(24):
            short='A' if step<12 else 'C'
            for name in (short,'B'):
                advance(model,live[name])
                if live[name]['step']==requests[name][1]:
                    result=finish(live.pop(name));assert result==controls[('cpu',name)],('interleaved',name)
                    report['rows'].append(dict(mode='interleaved_cpu',request=name,**result));atomic_json(args.output,report)
                    print('interleaved',name,'passes',flush=True)
            if step==11:
                assert 'B' in live and live['B']['step']==12
                live['C']=start(model,requests['C'][0],'cpu',args.direct,bounded=args.bounded,fused=args.fused,fused_bounded=args.fused_bounded,hot=args.hot,cold=args.cold)
        assert not live and len(report['rows'])==9
        report.update(passed=True,limitations='One model/seed, serialized default-stream fixed-length greedy requests; EOS ignored. Observers synchronize complete state. No asynchronous/threaded/batched-model serving, freegeneration quality, throughput/saturation, allocator peak or physicalSTT claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        for state in live.values():state['policy'].reset()
        atomic_json(args.output,report)


if __name__=='__main__':main()
