"""Reused development free-generation parity; no holdout tuning."""
import hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import check_reuse_integrity as driver
from bulk_promotion_policy import BulkPromotionPolicy
from cpu_phase_policy import CpuPhasePolicy
from shared_decode_transition import SharedAppendLease
class Control(driver.Trajectory,BulkPromotionPolicy):
    def __init__(self,**options):
        self.trajectory,self.frames=hashlib.sha256(),0
        super().__init__(**options)
class Candidate(driver.Trajectory,CpuPhasePolicy):
    counts={}
    def __init__(self,**options):
        self.trajectory,self.frames=hashlib.sha256(),0
        super().__init__(**options)
    def reset(self):
        type(self).counts=dict(cpu_promote_calls=self.cpu_promote_calls,cpu_append_calls=self.cpu_append_calls,cpu_demote_calls=self.cpu_demote_calls,cpu_fallback_steps=self.cpu_fallback_steps,cpu_evictions=sum(c.cpu_evictions for c in self.layer_caches.values()),cpu_reclaims=sum(c.cpu_reclaims for c in self.layer_caches.values()))
        super().reset()
def main():
    saved=driver.Control,driver.Candidate,driver.measure,driver.atomic_json
    leases=[]
    def measure(model,*args,**options):
        own=[];before=len(model._forward_pre_hooks),len(model._forward_hooks)
        if options['policy_cls'] is Candidate:
            def factory(**kw):
                p=Candidate(**kw);l=SharedAppendLease(model,p);own.append(l);leases.append(l);return p
            options['policy_cls']=factory
        try:
            result=saved[2](model,*args,**options)
            if own:
                result['append_fast_updates']=sum(l.fast_updates for l in own)
                result['append_fallback_updates']=sum(l.fallback_updates for l in own)
            return result
        finally:
            for l in own:l.close()
            assert before==(len(model._forward_pre_hooks),len(model._forward_hooks))
    def write(path,report):
        report.update(actual_control='ordinary_bulk_promotion',actual_candidate='cpu_phase_controller',owned_leases_closed=all(not l.handles and not l.pending for l in leases))
        saved[3](path,report)
    driver.Control,driver.Candidate,driver.measure,driver.atomic_json=Control,Candidate,measure,write
    try:driver.main()
    finally:
        for l in leases:l.close()
        driver.Control,driver.Candidate,driver.measure,driver.atomic_json=saved
if __name__=='__main__':main()
