"""Frozen full-model parity gate for early shared prefill."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import compare_reuse_visible as driver
from cpu_phase_policy import CpuPhasePolicy
from early_shared_cpu_phase import EarlySharedPolicy,SharedPrefillLease
from shared_decode_transition import SharedAppendLease
BASE=Path(__file__).resolve().parent
class Counts:
    phase_counts={}
    def reset(self):
        type(self).phase_counts=dict(early_shared_layers=getattr(self,'shared_prefill_layers',0),cpu_append_calls=self.cpu_append_calls,cpu_fallback_steps=self.cpu_fallback_steps)
        super().reset()
class Control(Counts,driver.RecordedNativePolicy,CpuPhasePolicy):pass
class Candidate(Counts,driver.RecordedNativePolicy,EarlySharedPolicy):pass

def main():
    context,decode=map(int,sys.argv[1:]);output=BASE/f'parity_{context}_{decode}.json'
    saved=(driver.RecordedNativePolicy,driver.trial.FusedBoundedPolicy,driver.trial.measure,driver.atomic_json,sys.argv)
    leases=[]
    def measure(model,*args,**kwargs):
        chosen=driver.trial.FusedBoundedPolicy;own=[]
        before=(len(model._forward_pre_hooks),len(model._forward_hooks))
        def factory(**options):
            policy=chosen(**options)
            if chosen is Candidate:own.append(SharedPrefillLease(model,policy))
            own.append(SharedAppendLease(model,policy));leases.extend(own);return policy
        driver.trial.FusedBoundedPolicy=factory
        try:
            row=saved[2](model,*args,**kwargs)
            row['phase_counts']=dict(chosen.phase_counts)
            assert row['phase_counts']['cpu_append_calls']==decode
            assert row['phase_counts']['cpu_fallback_steps']==0
            assert row['phase_counts']['early_shared_layers']==(32 if chosen is Candidate else 0)
            return row
        finally:
            for lease in own:lease.close()
            assert before==(len(model._forward_pre_hooks),len(model._forward_hooks))
            driver.trial.FusedBoundedPolicy=chosen
    def write(path,report):
        report.update(actual_control='cpu_phase',actual_candidate='early_shared_cpu_phase',owned_leases_closed=all(not v.handles for v in leases),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        saved[3](path,report)
    driver.RecordedNativePolicy=Candidate;driver.trial.FusedBoundedPolicy=Control;driver.trial.measure=measure;driver.atomic_json=write
    sys.argv=['check','--output',str(output),'--contexts',str(context),'--decode',str(decode),'--hot','649','--cold','1982']
    try:driver.main()
    finally:
        for lease in leases:lease.close()
        driver.RecordedNativePolicy,driver.trial.FusedBoundedPolicy,driver.trial.measure,driver.atomic_json,sys.argv=saved
if __name__=='__main__':main()
