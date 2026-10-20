"""Untimed fresh/legacy resource audit of the ordinary bulk/native CPU paths."""
import argparse
import hashlib
from pathlib import Path
import sys
import torch
import measure_fresh_process_memory as fresh
import measure_actual_reuse_memory as actual
import audit_exact_pool_resources as resources
import compare_cpu_batched as trial
import compare_bulk_promotion as comparison
from bulk_promotion_policy import BulkPromotionPolicy


def snapshot(model_ranges):
    segments=[s for s in torch.cuda.memory_snapshot() if s['device']==torch.cuda.current_device()]
    totals=dict(reserved_bytes=sum(s['total_size'] for s in segments),
        allocated_bytes=sum(s['allocated_size'] for s in segments),
        inactive_in_model_segments_bytes=0,inactive_in_model_only_segments_bytes=0)
    for segment in segments:
        active=[b for b in segment['blocks'] if b['state'].startswith('active')]
        tagged=[any(b['address']<=p<b['address']+b['size'] for p in model_ranges) for b in active]
        inactive=sum(b['size'] for b in segment['blocks'] if b['state']=='inactive')
        if any(tagged):totals['inactive_in_model_segments_bytes']+=inactive
        if tagged and all(tagged):totals['inactive_in_model_only_segments_bytes']+=inactive
    assert totals['reserved_bytes']==torch.cuda.memory_reserved()
    assert totals['allocated_bytes']==torch.cuda.memory_allocated()
    return dict(totals=totals,segments=segments,
        limitations='Public allocator snapshots; segment association is not causal attribution. Excludes non-Torch/driver allocations.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('fresh','legacy'),required=True)
    parser.add_argument('--arm',choices=('control','bulk'),default='bulk')
    parser.add_argument('--hot',type=int,required=True)
    parser.add_argument('--cold',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    holder={};weak={};handles=[];calls=[]
    driver=fresh if args.kind=='fresh' else trial
    original_loader=driver.AutoModelForCausalLM if args.kind=='fresh' else comparison.driver.AutoModelForCausalLM
    writer=fresh if args.kind=='fresh' else comparison.driver
    original_write=writer.atomic_json
    def alive():return bool(weak) and not torch.UntypedStorage._expired(weak['logits'])
    def before(model,positional,options):calls.append(dict(call=len(calls)+1,prefill_logits_alive_before_forward=alive()))
    def after(model,positional,options,out):
        storage=out.logits.untyped_storage()
        if len(calls)==1:weak['logits']=storage._weak_ref()
        calls[-1].update(logits_bytes=storage.nbytes(),shape=list(out.logits.shape))
    class Loader:
        @staticmethod
        def from_pretrained(*positional,**options):
            model=original_loader.from_pretrained(*positional,**options)
            roots={t.untyped_storage().data_ptr():t.untyped_storage().nbytes()
                   for t in list(model.parameters())+list(model.buffers()) if t.device.type=='cuda'}
            holder.update(constant_model_storage_bytes=sum(roots.values()),model_ranges=roots)
            torch.cuda.synchronize();holder['loaded_model_snapshot']=snapshot(roots)
            handles.extend((model.register_forward_pre_hook(before,with_kwargs=True),model.register_forward_hook(after,with_kwargs=True)))
            return model
    def write(path,report):
        report.update(resource_audit_kind=args.kind,actual_resource_arm=args.arm if args.kind=='fresh' else 'paired_control_bulk',
            resource_model=holder,logits_lifetime_rows=calls,
            resource_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            observer_limitations='Primitive public snapshots and weak storage handles retain no GPU payload/model owners. Observer timings invalid; no superiority/quality/physicalSTT claim. Other host/driver scopes excluded.')
        original_write(path,report)
    saved_argv=sys.argv
    if args.kind=='fresh':
        original_policy,original_memory=actual.NativeReusePolicy,fresh.memory
        if args.arm=='bulk':actual.NativeReusePolicy=BulkPromotionPolicy
        def memory(stage):
            result=original_memory(stage)
            if stage=='generation_complete':holder['decode_final_snapshot']=snapshot(holder['model_ranges'])
            return result
        fresh.AutoModelForCausalLM,fresh.memory=Loader,memory
        run=actual.main
        sys.argv=['audit','--arm','native_reuse','--hot',str(args.hot),'--cold',str(args.cold),'--output',str(args.output)]
    else:
        original_comparison=resources.comparison.main
        resources.comparison.main=comparison.main
        comparison.driver.AutoModelForCausalLM=Loader
        run=resources.main
        sys.argv=['audit','--contexts','16384','--decode','12','--hot',str(args.hot),'--cold',str(args.cold),'--output',str(args.output)]
    writer.atomic_json=write
    try:run()
    finally:
        sys.argv=saved_argv;writer.atomic_json=original_write
        for handle in handles:handle.remove()
        if weak:torch.UntypedStorage._free_weak_ref(weak['logits'])
        if args.kind=='fresh':
            actual.NativeReusePolicy=original_policy
            fresh.AutoModelForCausalLM,fresh.memory=original_loader,original_memory
        else:
            resources.comparison.main=original_comparison
            comparison.driver.AutoModelForCausalLM=original_loader


if __name__=='__main__':main()
