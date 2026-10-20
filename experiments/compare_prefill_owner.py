"""Same forward-leased raw-payload policy; initial hot backing alone differs."""
import hashlib
from pathlib import Path
import compare_reuse_visible as driver
from compare_forward_owner import RecordedForwardPolicy
from forward_owner_release import OwnerRelease
from prefill_owner_share import PrefillSharedPolicy


class RecordedSharedPolicy(RecordedForwardPolicy,PrefillSharedPolicy):
    last_prefill_shared=0

    def reset(self):
        type(self).last_prefill_shared=self.prefill_shared_layers
        super().reset()


def main():
    trial=driver.trial
    previous=trial.FusedBoundedPolicy,driver.RecordedNativePolicy,trial.measure,driver.atomic_json
    def measure(model,*args,**kwargs):
        chosen=trial.FusedBoundedPolicy;leases=[];result=None
        before=(len(model._forward_pre_hooks),len(model._forward_hooks))
        def factory(**options):
            policy=chosen(**options)
            leases.append(OwnerRelease(model,policy))
            return policy
        trial.FusedBoundedPolicy=factory
        try:
            result=previous[2](model,*args,**kwargs)
            result['forward_owner_counts']=dict(released_forwards=sum(l.released_forwards for l in leases),
                restored_forwards=sum(l.restored_forwards for l in leases),
                released_layers=sum(l.released_layers for l in leases))
            if chosen is RecordedSharedPolicy:result['prefill_shared_layers']=chosen.last_prefill_shared
            return result
        finally:
            for lease in leases:lease.close()
            trial.FusedBoundedPolicy=chosen
            assert before==(len(model._forward_pre_hooks),len(model._forward_hooks))
            if result is not None:result['owner_hooks_removed']=True
    def write(path,report):
        report.update(actual_control='research_forward_owner_release',actual_candidate='prefill_owner_share',
            prefill_owner_share=True,prefill_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            prefill_helper_sha256=hashlib.sha256(Path(__file__).with_name('prefill_owner_share.py').read_bytes()).hexdigest(),
            prefill_limits='Same scoped serial OwnerRelease and RawInputPolicy in both arms; only initial hot backing differs. Public whole-decode/host envelope, not complete physical resource equality. Mandatory timing/generation/lifetime/serving gates remain; no ordinary/default change.')
        previous[3](path,report)
    trial.FusedBoundedPolicy=RecordedForwardPolicy;driver.RecordedNativePolicy=RecordedSharedPolicy
    trial.measure=measure;driver.atomic_json=write
    try:driver.main()
    finally:trial.FusedBoundedPolicy,driver.RecordedNativePolicy,trial.measure,driver.atomic_json=previous


if __name__=='__main__':main()
