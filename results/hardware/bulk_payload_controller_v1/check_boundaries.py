"""Frozen full-model parity gate for early shared prefill."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import compare_reuse_visible as driver
from cpu_phase_policy import CpuPhasePolicy
from bulk_payload_policy import BulkPayloadPolicy
from shared_decode_transition import SharedAppendLease
BASE=Path(__file__).resolve().parent
OPTIONS={}
CASES=dict(exclusive=dict(inclusive=False),floor_over_capacity=dict(stt_live_floor=256),write_aware_fallback=dict(write_aware_lambda=.125))
class Counts:
    phase_counts={}
    def __init__(self,**options):super().__init__(**(options|OPTIONS))
    def reset(self):
        type(self).phase_counts=dict(bulk_steps=getattr(self,'bulk_payload_steps',0),bulk_fallback_steps=getattr(self,'bulk_payload_fallback_steps',0),cpu_append_calls=self.cpu_append_calls,cpu_fallback_steps=self.cpu_fallback_steps)
        super().reset()
class Control(Counts,driver.RecordedNativePolicy,CpuPhasePolicy):pass
class Candidate(Counts,driver.RecordedNativePolicy,BulkPayloadPolicy):pass

def main(case):
    global OPTIONS
    OPTIONS=CASES[case];context,decode=4096,50;output=BASE/('boundary_'+case+'.json')
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
            counts=row['phase_counts']
            if case=='write_aware_fallback':
                assert counts['cpu_append_calls']==0 and counts['cpu_fallback_steps']==50
            else:assert counts['cpu_append_calls']==50 and counts['cpu_fallback_steps']==0
            if chosen is Candidate:
                assert counts['bulk_steps']+counts['bulk_fallback_steps']==50
                if case in ['exclusive','write_aware_fallback']:assert counts['bulk_steps']==0
            return row
        finally:
            for lease in own:lease.close()
            assert before==(len(model._forward_pre_hooks),len(model._forward_hooks))
            driver.trial.FusedBoundedPolicy=chosen
    def write(path,report):
        report.update(actual_control='cpu_phase',actual_candidate='bulk_payload_controller',owned_leases_closed=all(not v.handles for v in leases),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        saved[3](path,report)
    driver.RecordedNativePolicy=Candidate;driver.trial.FusedBoundedPolicy=Control;driver.trial.measure=measure;driver.atomic_json=write
    sys.argv=['check','--output',str(output),'--contexts',str(context),'--decode',str(decode),'--hot','649','--cold','128']
    try:driver.main()
    finally:
        for lease in leases:lease.close()
        driver.RecordedNativePolicy,driver.trial.FusedBoundedPolicy,driver.trial.measure,driver.atomic_json,sys.argv=saved
if __name__=='__main__':
    for case in CASES:main(case)
