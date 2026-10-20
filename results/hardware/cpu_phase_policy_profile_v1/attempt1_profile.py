"""Diagnostic attribution of existing CPU phase work; no timing/adoption gate."""
import cProfile,hashlib,importlib.util,json,pstats,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from cpu_phase_policy import CpuPhasePolicy
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
profiles={name:cProfile.Profile() for name in ['policy','scores']}
class Profiled(CpuPhasePolicy):
    def __call__(self,*args,**kwargs):return profiles['policy'].runcall(super().__call__,*args,**kwargs)
    def update_scores(self,*args,**kwargs):return profiles['scores'].runcall(super().update_scores,*args,**kwargs)

def main():
    spec=importlib.util.spec_from_file_location('frozen_resource_caller',ROOT/'results/hardware/prompt_boundary_comparison_v1/audit_cpu.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    driver.BASE=BASE;driver.ReservedAppendPolicy=Profiled
    try:driver.main()
    finally:
        for name,profile in profiles.items():
            profile.dump_stats(str(BASE/(name+'.prof')))
            with (BASE/(name+'.txt')).open('w') as stream:pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(55)
    ref=json.loads((ROOT/'results/hardware/prompt_boundary_comparison_v1/cpu.json').read_text());actual=json.loads((BASE/'cpu.json').read_text())
    for name in ['predicted_token_ids','state_sha256','counters','prototype_counts']:assert actual[name]==ref[name],name
    assert actual['cpu_append_calls']==50 and actual['cpu_fallback_steps']==0 and actual['owned_pinned_after_cleanup_bytes']==0
    (BASE/'verified_profile.json').write_text(json.dumps(dict(verified=True,full_16K50_replay_exact=True,no_fallback=True,profile_instrumented=True,valid_timing_gate=False,full_goal_complete=False,evidence_sha256={p.name:sha(p) for p in BASE.iterdir() if p.suffix in ['.json','.prof','.txt','.py']},reference_sha256=sha(ROOT/'results/hardware/prompt_boundary_comparison_v1/cpu.json')),indent=2)+'\n')
if __name__=='__main__':main()
