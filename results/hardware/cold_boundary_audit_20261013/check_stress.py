"""Mandatory wider-context and cold-slot recycling parity."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import compare_reuse_visible as driver
from cold_boundary_policy import ColdBoundaryPolicy
from bulk_payload_policy import BulkPayloadPolicy
from shared_decode_transition import SharedAppendLease
BASE=Path(__file__).resolve().parent
class Counts:
    phase_counts={}
    def reset(self):
        type(self).phase_counts=dict(cold_steps=getattr(self,'cold_boundary_steps',0),cold_fallback_calls=getattr(self,'cold_boundary_fallback_calls',0),bulk_steps=getattr(self,'bulk_payload_steps',0),bulk_fallback_steps=getattr(self,'bulk_payload_fallback_steps',0),cpu_append_calls=self.cpu_append_calls,cpu_fallback_steps=self.cpu_fallback_steps,cpu_reclaims=sum(c.cpu_reclaims for c in self.layer_caches.values()))
        super().reset()
class Control(Counts,driver.RecordedNativePolicy,BulkPayloadPolicy):pass
class Candidate(Counts,driver.RecordedNativePolicy,ColdBoundaryPolicy):pass

def main():
    context,decode,cold=map(int,sys.argv[1:]);output=BASE/f'stress_{context}_{decode}_{cold}.json'
    saved=(driver.RecordedNativePolicy,driver.trial.FusedBoundedPolicy,driver.trial.measure,driver.atomic_json,sys.argv)
    leases=[]
    def measure(model,*args,**kwargs):
        chosen=driver.trial.FusedBoundedPolicy;own=[]
        before=(len(model._forward_pre_hooks),len(model._forward_hooks))
        def factory(**options):
            policy=chosen(**options)
            
            own.append(SharedAppendLease(model,policy));leases.extend(own);return policy
        driver.trial.FusedBoundedPolicy=factory
        try:
            row=saved[2](model,*args,**kwargs)
            row['phase_counts']=dict(chosen.phase_counts)
            assert row['phase_counts']['cpu_append_calls']==decode
            assert row['phase_counts']['cpu_fallback_steps']==0
            if chosen is Candidate:
                assert row['phase_counts']['bulk_steps']+row['phase_counts']['bulk_fallback_steps']==decode
                if context==8192:assert row['phase_counts']['bulk_steps']==decode
            else:assert row['phase_counts']['bulk_steps']+row['phase_counts']['bulk_fallback_steps']==decode
            if chosen is Candidate and context==8192:assert row['phase_counts']['cold_steps']==decode
            return row
        finally:
            for lease in own:lease.close()
            assert before==(len(model._forward_pre_hooks),len(model._forward_hooks))
            driver.trial.FusedBoundedPolicy=chosen
    def write(path,report):
        report.update(actual_control='bulk_payload_controller',actual_candidate='cold_boundary_controller',owned_leases_closed=all(not v.handles for v in leases),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        saved[3](path,report)
    driver.RecordedNativePolicy=Candidate;driver.trial.FusedBoundedPolicy=Control;driver.trial.measure=measure;driver.atomic_json=write
    sys.argv=['check','--output',str(output),'--contexts',str(context),'--decode',str(decode),'--hot','649','--cold',str(cold)]
    try:driver.main()
    finally:
        for lease in leases:lease.close()
        driver.RecordedNativePolicy,driver.trial.FusedBoundedPolicy,driver.trial.measure,driver.atomic_json,sys.argv=saved
if __name__=='__main__':
    plan=json.loads((BASE/'stress_plan.json').read_text())
    for case in plan['cases']:
        sys.argv=['stress',str(case['context']),str(case['decode']),str(case['cold'])];main()
