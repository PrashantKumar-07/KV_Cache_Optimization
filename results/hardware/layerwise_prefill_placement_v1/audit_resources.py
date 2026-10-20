"""Unchanged fresh caller with scoped layerwise cache placement only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments'))
import audit_bulk_resources as audit
from bulk_promotion_policy import BulkPromotionPolicy
from candidate import EarlyBulkPolicy, PrefillLease, PLACEMENT_AST_SHA256


def counts(policy):
    caches=policy.layer_caches.values()
    result={name:sum(getattr(c,name,0) for c in caches) for name in
            ('host_read_bytes','host_write_bytes','host_initial_write_bytes')}
    result.update(prefill_shared_layers=0,raw_bound_layers=0,raw_compactions=0,
        native_h2d_calls=getattr(policy.manager,'h2d_calls',0),
        native_d2h_calls=getattr(policy.manager,'d2h_calls',0))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('control','candidate'),required=True)
    parser.add_argument('--hot',type=int,default=991);parser.add_argument('--cold',type=int,default=1090)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    original=(audit.fresh.AutoModelForCausalLM,audit.BulkPromotionPolicy,
              audit.fresh.atomic_json,audit.fresh.main,sys.argv)
    models=[];leases=[];baseline=[];final={}
    class Loader:
        @staticmethod
        def from_pretrained(*positional,**options):
            model=original[0].from_pretrained(*positional,**options)
            models.append(model);baseline.append((len(model._forward_pre_hooks),len(model._forward_hooks)))
            return model
    class Capture:
        def reset(self):
            final.update(prototype_counts=counts(self),
                placement_rows=list(getattr(self,'placement_rows',[])),
                placed_layers=list(getattr(self,'placed_layers',[])),
                query_map_empty=not self.prompt_q_by_layer)
            super().reset()
    class Control(Capture,BulkPromotionPolicy):pass
    class Candidate(Capture,EarlyBulkPolicy):pass
    def factory(**options):
        policy=(Candidate if args.arm=='candidate' else Control)(**options)
        if args.arm=='candidate':leases.append(PrefillLease(models[0],policy))
        return policy
    def write(path,report):
        for lease in leases:lease.close()
        assert (len(models[0]._forward_pre_hooks),len(models[0]._forward_hooks))==(
            baseline[0][0]+1,baseline[0][1]+1),'Own prefill hooks still installed'
        report.update(final,layerwise_arm=args.arm,placement_ast_sha256=PLACEMENT_AST_SHA256,
            layerwise_hooks_removed=True,layerwise_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            layerwise_limits='Unchanged ordinary fresh caller/reset boundaries; one-model fixed teacher stream. '
                'Exact early initial placement only; full raw KV returned to prefill attention. '
                'Untimed resource/preservation gate; no quality superiority/resource equality/physicalSTT/adoption claim.')
        original[2](path,report)
    def run():
        saved=sys.argv;sys.argv=[*saved,'--prefill-last-logits']
        try:original[3]()
        finally:sys.argv=saved
    audit.fresh.AutoModelForCausalLM=Loader;audit.BulkPromotionPolicy=factory
    audit.fresh.atomic_json=write;audit.fresh.main=run
    sys.argv=['audit','--kind','fresh','--arm','bulk','--hot',str(args.hot),'--cold',str(args.cold),
              '--output',str(args.output)]
    try:audit.main()
    finally:
        for lease in leases:lease.close()
        if models:assert baseline[0]==(len(models[0]._forward_pre_hooks),len(models[0]._forward_hooks))
        (audit.fresh.AutoModelForCausalLM,audit.BulkPromotionPolicy,
         audit.fresh.atomic_json,audit.fresh.main,sys.argv)=original
        cleanup=args.output.with_name(args.output.stem+'_cleanup.json')
        assert not cleanup.exists()
        cleanup.write_text(json.dumps(dict(root_hooks_restored=True,
            leases_closed=all(not lease.handles for lease in leases)),indent=2)+'\n')


if __name__=='__main__':main()
