"""Untimed dispatch/storage-lifetime trace of the composed CPU policy decode."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from transformers import AutoModelForCausalLM, AutoTokenizer
import audit_snap_tier_envelope as envelope
import fused_hot_compaction as compaction
from fused_bounded_pool import FusedBoundedPoolPolicy
from provenance import atomic_json, runtime_identity


def is_kv(tensor):
    if tensor.dtype not in (torch.bfloat16,torch.float32) or tensor.shape[-1:]!=(128,):
        return False
    shape=tensor.shape
    return ((len(shape)==3 and shape[0]==8) or
        (len(shape)==4 and ((shape[0]==1 and shape[1]==8) or
                           (shape[0]==2 and shape[1]%8==0))) or
        (len(shape)==5 and shape[0]==2 and shape[2]==8))


class StorageTrace(TorchDispatchMode):
    """Weak StorageImpl handles follow aliases without retaining KV payloads."""
    def __init__(self):
        super().__init__()
        self.stage='decode_update'
        self.allocations=[];self.samples=[];self.live={}

    def track(self,value,origin):
        for tensor in tree_flatten(value)[0]:
            if not isinstance(tensor,torch.Tensor) or not is_kv(tensor):continue
            storage=tensor.untyped_storage();key=storage._cdata
            if key in self.live:continue
            index=len(self.allocations)
            self.allocations.append(dict(id=index,origin=origin,device=str(tensor.device),
                bytes=storage.nbytes(),shape=list(tensor.shape),dtype=str(tensor.dtype),
                birth_event=len(self.samples),end_event=None))
            self.live[key]=(storage._weak_ref(),type(storage),index)

    def sample(self,operation):
        totals={};count=0
        for key,(handle,cls,index) in list(self.live.items()):
            if cls._expired(handle):
                self.allocations[index]['end_event']=len(self.samples)
                cls._free_weak_ref(handle);del self.live[key]
            else:
                item=self.allocations[index];count+=1
                totals[item['device']]=totals.get(item['device'],0)+item['bytes']
        self.samples.append(dict(event=len(self.samples),stage=self.stage,operation=operation,
            known_live_kv_storage_bytes=totals,known_live_kv_total_bytes=sum(totals.values()),
            live_storage_count=count))

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        self.track((args,result),str(func))
        self.sample(str(func))
        return result

    @contextmanager
    def phase(self,name):
        previous=self.stage;self.stage=name;self.sample('phase_enter')
        try:yield
        finally:self.sample('phase_exit');self.stage=previous

    def close(self):
        for handle,cls,_ in self.live.values():cls._free_weak_ref(handle)
        self.live.clear()


class ObservedPolicy(FusedBoundedPoolPolicy):
    traces=[]
    active=None

    def _decode_step(self,past):
        trace=StorageTrace();type(self).active=trace
        caches=list(self.layer_caches.values())
        trace.track([past,self.manager.pool,
            [(c.sram_k,c.sram_v) for c in caches]],'decode_entry')
        trace.sample('decode_entry')
        try:
            with trace:
                result=super()._decode_step(past)
                trace.track(result,'decode_return');trace.sample('decode_return')
            type(self).traces.append(dict(step=self.decode_step_count,
                allocations=trace.allocations,samples=trace.samples))
            return result
        finally:
            type(self).active=None;trace.close()

    def _prepare_decode_reads(self):
        with type(self).active.phase('promotion_prepare'):
            return super()._prepare_decode_reads()

    def _prepare_visible_reads(self,selections):
        if type(self).active is None:return super()._prepare_visible_reads(selections)
        with type(self).active.phase('visible_reads'):
            return super()._prepare_visible_reads(selections)

    def _visible_tuple(self,past):
        if type(self).active is None:return super()._visible_tuple(past)
        with type(self).active.phase('visible_assembly'):
            result=super()._visible_tuple(past)
            type(self).active.track(result,'visible_return')
            type(self).active.sample('visible_return')
            return result


class ObservedKernel:
    def __init__(self,module):self.module=module

    def compact(self,*args):
        trace=ObservedPolicy.active
        with trace.phase('fused_launch'):
            trace.track(args[3],'native_output');trace.sample('native_launch')
            return self.module.compact(*args)


def self_check():
    trace=StorageTrace()
    tensor=torch.empty((8,2,128),dtype=torch.bfloat16);view=tensor[:,1:]
    trace.track([tensor,view],'fixture');trace.sample('owner_and_view')
    assert len(trace.allocations)==1 and trace.samples[-1]['known_live_kv_total_bytes']==4096
    del tensor;gc.collect();trace.sample('view_only')
    assert trace.samples[-1]['known_live_kv_total_bytes']==4096
    del view;gc.collect();trace.sample('released')
    assert trace.samples[-1]['known_live_kv_total_bytes']==0
    assert trace.allocations[0]['end_event']==2
    assert not is_kv(torch.empty(128,dtype=torch.float32))
    assert not is_kv(torch.empty((1,32,2,128),dtype=torch.bfloat16))
    trace.close()
    return dict(storage_alias_counted_once=True,view_keeps_storage_alive=True,
        weak_handle_does_not_retain_payload=True,metadata_and_query_shapes_excluded=True)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--self-check-only',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    sources=list(Path('experiments').glob('*.py'))+list(Path('src').glob('*.py'))+[
        Path('experiments/native_hot_compact.cpp'),Path('experiments/native_hot_compact.cu')]
    report=dict(passed=False,runtime=runtime_identity(),self_check=self_check(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    original_cls,original_kernel=envelope.DirectScorePolicy,compaction.kernel
    try:
        if args.self_check_only:
            report['passed']=True;return
        module=original_kernel();ObservedPolicy.traces=[]
        report['binary_sha256']=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        envelope.DirectScorePolicy=ObservedPolicy
        compaction.kernel=lambda:ObservedKernel(module)
        prompt=Path('results/remediation_v3/latency_prompt.txt')
        teacher=Path('results/shadow_ablation_full/fixed_stream_20260929.json')
        stream=json.loads(teacher.read_text())['config']['teacher_token_ids'][:50]
        assert len(stream)==50
        torch.manual_seed(0);model_path='../models/mistral-7b-instruct'
        model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(model_path)
        ids=tokenizer(prompt.read_text(),return_tensors='pt').input_ids[:,:16384].to('cuda')
        assert ids.shape==(1,16384)
        report.update(model=model_path,seed=0,context=16384,decode=50,gpu=torch.cuda.get_device_name(),
            prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
            teacher_source_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
            input_token_ids=ids.cpu().tolist()[0],teacher_token_ids=stream)
        report['result']=envelope.run(model,ids,stream,'cpu_direct')
        report['traces']=ObservedPolicy.traces
        assert len(report['traces'])==50
        prior=json.loads(Path('results/baseline_fidelity/snap_fused_bounded_envelope_20261002/audit.json').read_text())
        for field in ('predicted_token_ids','counters','h2d_payload_bytes','d2h_payload_bytes'):
            assert report['result'][field]==prior['result'][field],field
        peaks={}
        for trace in report['traces']:
            for sample in trace['samples']:
                key=sample['stage']
                if key not in peaks or sample['known_live_kv_total_bytes']>peaks[key]['known_live_kv_total_bytes']:
                    peaks[key]=dict(step=trace['step'],**sample)
        report.update(passed=True,phase_peaks=peaks,policy_preserved=True,
            limitations='Untimed default-stream policy trace; BF16/F32 tensors of grouped-KV shapes, complete storage behind views, pinned request root and weak C++ storage lifetimes. Attention/model-forward temporaries, untracked prior locals, metadata, retained allocator blocks and released-but-stream-pending buffers excluded. Shape classification may include KV-sized policy scratch. This is known live storage, not total physical peak, exact resource equality or a speed claim. Private weak-storage APIs restricted to measured Torch environment.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        envelope.DirectScorePolicy=original_cls;compaction.kernel=original_kernel
        ObservedPolicy.active=None;atomic_json(args.output,report)


if __name__=='__main__':main()
