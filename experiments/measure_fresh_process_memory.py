"""Fresh-process GPU/RSS and explicitly owned pinned-buffer memory audit."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import weakref
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from compare_cpu_batched import digest
from batched_score_policy import ScoreBatchPolicy
from direct_shared_pool import DirectScorePolicy
from bounded_pinned_pool import BoundedPoolPolicy
from evaluate_query_selection import PromptQueryCapture
from longbench_eval import TieredKVPolicy, _attn_impl, apply_tuple_kv_inplace, to_tuple_kv, prefill_logit_kwargs
from provenance import atomic_json, runtime_identity


def proc_memory():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS','VmHWM','RssAnon','RssFile','VmLck'):
            values[key+'_bytes'] = int(value.split()[0])*1024
    return values


def memory(stage):
    return dict(stage=stage,**proc_memory(),gpu_allocated_bytes=torch.cuda.memory_allocated(),
        gpu_reserved_bytes=torch.cuda.memory_reserved(),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())


class PinnedRoots:
    """Track explicit root tensor ownership, not CachingHostAllocator reservations."""
    def __init__(self):
        self.original = torch.empty; self.current = self.peak = 0; self.events = []

    def empty(self,*args,**kwargs):
        tensor = self.original(*args,**kwargs)
        if kwargs.get('pin_memory',False):
            self.track(tensor)
        return tensor

    def track(self,tensor):
        assert tensor.is_pinned()
        size = tensor.untyped_storage().nbytes()
        self.current += size; self.peak = max(self.peak,self.current)
        self.events.append(dict(kind='allocate',bytes=size,owned_bytes=self.current,shape=list(tensor.shape)))
        weakref.finalize(tensor,self.release,size)

    def release(self,size):
        self.current -= size
        self.events.append(dict(kind='release',bytes=size,owned_bytes=self.current))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('gpu_query','cpu_query','cpu_direct','cpu_bounded'),required=True)
    parser.add_argument('--hot',type=int,default=1024)
    parser.add_argument('--cold',type=int,default=2048)
    parser.add_argument('--prefill-last-logits',action='store_true',
        help='Compute only last prefill logits; requires native logits_to_keep support')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(f'Preserve evidence: {args.output}')
    sources=('experiments/measure_fresh_process_memory.py','experiments/compare_cpu_batched.py',
        'experiments/evaluate_query_selection.py','experiments/batched_score_policy.py',
        'experiments/cpu_batched_cache.py','experiments/cpu_slot_cache.py','experiments/direct_shared_pool.py','experiments/bounded_pinned_pool.py')
    report=dict(passed=False,pid=os.getpid(),arm=args.arm,hot=args.hot,cold=args.cold,
        prefill_last_logits=args.prefill_last_logits,runtime=runtime_identity(),boundaries=[],rss_samples=[],
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        host_allocator_stats_available=hasattr(torch.cuda.memory,'host_memory_stats'),
        internal_host_cache_flush_available=hasattr(torch._C,'_host_emptyCache'))
    tracker=PinnedRoots(); stop=threading.Event(); sampler=None;policy=capture=past=out=None
    try:
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        prompt=Path('results/remediation_v3/latency_prompt.txt').read_text()
        teacher=Path('results/shadow_ablation_full/fixed_stream_20260929.json')
        tokens=json.loads(teacher.read_text())['config']['teacher_token_ids'][:50];assert len(tokens)==50
        ids=tokenizer(prompt,truncation=False,return_tensors='pt').input_ids[:,:16384].to('cuda')
        assert ids.shape==(1,16384)
        report.update(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),teacher_source_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
            teacher_token_ids=tokens,input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),gpu=torch.cuda.get_device_name())
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        report['boundaries'].append(memory('loaded_model'))
        policy={'gpu_query':TieredKVPolicy,'cpu_query':ScoreBatchPolicy,'cpu_direct':DirectScorePolicy,'cpu_bounded':BoundedPoolPolicy}[args.arm](sram_budget=args.hot,
            stt_budget=args.cold,start_size=4,recent_size=128,kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
        policy.prompt_query_score_mode='chunked';policy.prompt_query_chunk=128
        capture=PromptQueryCapture(model,policy)
        torch.empty=tracker.empty
        started=time.monotonic()
        def sample():
            while not stop.is_set():
                report['rss_samples'].append(dict(elapsed_s=time.monotonic()-started,**proc_memory()))
                stop.wait(.02)
        sampler=threading.Thread(target=sample,daemon=True);sampler.start()
        prefill_options=prefill_logit_kwargs(model,args.prefill_last_logits)
        with _attn_impl(model,'sdpa'):
            out=model(ids,use_cache=True,position_ids=torch.arange(16384,device='cuda')[None],**prefill_options)
        past=out.past_key_values
        apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
        torch.cuda.synchronize()
        report['boundaries'].append(memory('prefill_compressed'))
        capture.close();capture=None
        del out  # Prefill logits are unused by the fixed teacher stream.
        predictions=[]
        for step,token in enumerate(tokens):
            out=model(torch.tensor([[token]],device='cuda'),past_key_values=past,use_cache=True,
                position_ids=torch.tensor([[16384+step]],device='cuda'),output_attentions=True)
            past=out.past_key_values;tup=to_tuple_kv(past)
            policy.update_scores(tup,out.attentions);apply_tuple_kv_inplace(past,policy(tup))
            predictions.append(out.logits[:,-1:].argmax(-1).item())
            del tup
            torch.cuda.synchronize()
            report['boundaries'].append(memory(f'decode_{step+1}'))
        stop.set();sampler.join();sampler=None
        report.update(predicted_token_ids=predictions,measurement_final=memory('generation_complete'),
            owned_pinned_current_bytes=tracker.current,owned_pinned_peak_bytes=tracker.peak,
            occupied_host_bytes=sum(getattr(c,'occupied_host_bytes',0) for c in policy.layer_caches.values()),
            reserved_host_bytes=sum(getattr(c,'reserved_host_bytes',0) for c in policy.layer_caches.values()),
            counters=policy._live_totals())
        # Integrity hash after phase measurements: canonical CPU inspection is diagnostic work.
        report['state_sha256']=digest(policy.layer_caches)
        policy.reset();policy=None;past=out=None;gc.collect();torch.cuda.synchronize()
        report['cleanup']=[memory('after_reset_gc')]
        assert tracker.current==0,'Pinned root tensors still owned after cleanup'
        if report['internal_host_cache_flush_available']:
            torch._C._host_emptyCache()
            report['cleanup'].append(memory('after_host_cache_flush'))
        torch.cuda.empty_cache();report['cleanup'].append(memory('after_gpu_cache_flush'))
        report.update(passed=True,owned_pinned_after_cleanup_bytes=tracker.current,pinned_root_events=tracker.events,
            limitations='Fresh one-arm process/one model/seed/fixed teacher stream. RSS samples lowerbound; VmHWM includes model load. Explicit requested pinned root bytes exclude allocator rounding/retained blocks; no public pinned allocator stats in Torch2.5. Hostflush RSS difference descriptive, not exact pinned cache size. Canonical state hash after measured generation; GPU peaks include model/query scoring/attention. Unequal host resources; NarrativeQA NI remains unproven, no promotion/fairbaseline/speed/physicalSTT claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        stop.set()
        if sampler is not None:sampler.join()
        if capture is not None:capture.close()
        if policy is not None:policy.reset()
        torch.empty=tracker.original
        atomic_json(args.output,report)
    print(args.arm,'passes',report['owned_pinned_peak_bytes'],report['measurement_final']['gpu_peak_allocated_bytes'],flush=True)


if __name__=='__main__':main()
