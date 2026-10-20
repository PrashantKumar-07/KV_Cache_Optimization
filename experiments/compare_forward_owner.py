"""Fixed research raw-reuse control versus temporary forward-owner release."""
import hashlib
from pathlib import Path
import compare_raw_input_reuse as comparison
from forward_owner_release import ForwardOwnerPolicy,OwnerRelease


class RecordedForwardPolicy(comparison.RecordedRawPolicy,ForwardOwnerPolicy):
    pass


def main():
    trial=comparison.comparison.driver.trial
    original=(comparison.RecordedRawPolicy,comparison.comparison.RecordedBulkPolicy,
              trial.measure,comparison.comparison.driver.atomic_json)
    summaries=[]
    def measure(model,*args,**kwargs):
        chosen=trial.FusedBoundedPolicy
        leases=[]
        if chosen is RecordedForwardPolicy:
            def factory(**options):
                policy=chosen(**options)
                leases.append(OwnerRelease(model,policy))
                return policy
            trial.FusedBoundedPolicy=factory
        try:
            result=original[2](model,*args,**kwargs)
            if leases:
                result['forward_owner_counts']=dict(released_forwards=sum(v.released_forwards for v in leases),
                    restored_forwards=sum(v.restored_forwards for v in leases),
                    released_layers=sum(v.released_layers for v in leases))
                summaries.append(result['forward_owner_counts'])
            return result
        finally:
            for lease in leases:lease.close()
            trial.FusedBoundedPolicy=chosen
    def write(path,report):
        report.update(actual_control='research_raw_input_reuse',actual_candidate='forward_owner_release',
            forward_owner_release=True,forward_owner_summaries=summaries,
            owner_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            owner_helper_sha256=hashlib.sha256(Path(__file__).with_name('forward_owner_release.py').read_bytes()).hexdigest(),
            owner_limits='Scoped serial Mistral/DynamicLayer owner release; standardTorch payload prototype unchanged. No ordinary/default/physicalSTT claim; macro timing and mandatory exception/lifetime/generation/recycled-request gates remain.')
        original[3](path,report)
    comparison.RecordedRawPolicy=RecordedForwardPolicy
    comparison.comparison.RecordedBulkPolicy=original[0]
    trial.measure=measure;comparison.comparison.driver.atomic_json=write
    try:comparison.main()
    finally:
        comparison.RecordedRawPolicy,comparison.comparison.RecordedBulkPolicy,trial.measure,comparison.comparison.driver.atomic_json=original


if __name__=='__main__':main()
