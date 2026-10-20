"""Keep decode memory scopes explicit; stop on the strict GPU-only guard."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def main():
    path=BASE/'verified_guard_review.json';assert not path.exists()
    proof=read(BASE/'verified_resources_shared.json')
    assert proof['verified'] and proof['memory_primary_passed']
    assert all(sha(BASE/p)==v for p,v in proof['evidence_sha256'].items())
    a=read(BASE/'control_shared.json');b=read(BASE/'candidate_shared.json')
    constant=a['resource_model']['constant_model_storage_bytes']
    assert constant==b['resource_model']['constant_model_storage_bytes']
    host=a['owned_host_allocator_block_peak_bytes']
    assert host==b['owned_host_allocator_block_peak_bytes']==256*2**20
    gpu_a=a['gpu_phase_peaks']['decode_peak']-constant
    gpu_b=b['gpu_phase_peaks']['decode_peak']-constant
    change=gpu_b/gpu_a-1
    assert gpu_a/2**20==proof['control_active_decode_mib']
    assert gpu_b/2**20==proof['candidate_active_decode_mib']
    assert change>.02
    plan=read(BASE/'plan.json')
    assert 'decode active/reserved peaks<=2% worse' in plan['timing']
    result=dict(verified=True,memory_primary_passed=True,
        control_prefill_gb=a['gpu_phase_peaks']['prefill_peak']/1e9,
        candidate_prefill_gb=b['gpu_phase_peaks']['prefill_peak']/1e9,
        prefill_reduction_pct=100*proof['relative_prefill_reduction'],
        gpu_only_control_decode_active_mib=gpu_a/2**20,
        gpu_only_candidate_decode_active_mib=gpu_b/2**20,
        gpu_only_decode_active_change_pct=100*change,
        combined_control_decode_envelope_mib=(gpu_a+host)/2**20,
        combined_candidate_decode_envelope_mib=(gpu_b+host)/2**20,
        combined_decode_envelope_change_pct=100*((gpu_b+host)/(gpu_a+host)-1),
        guard_passed=False,disposition='Research tradeoff; stop expansion, no ordinary integration.',
        reason='Preregistered decode active/reserved<=2% guard already fails in the untimed pair when active means GPU-only allocation excluding constant model storage. Plan did not explicitly exclude that stricter scope; do not substitute the combined GPU+host or model-included denominator to claim a pass.',
        second_geometry_generation_timing_stopped=True,all_owned_jobs_finished=True,
        ordinary_source_changed=False,full_goal_complete=False,
        next='Return to actual resource matched quality/latency P0 work; freeze this tradeoff instead of retrying it for a better guard result.',
        limits='One model/seed/16K50; CPU full-logit/cache equivalence and GPU teacher replay. No GPU generation/timing proof, adoption, speed, complete512MiB, quality superiority or physicalSTT claim.',
        evidence_sha256={p:sha(BASE/p) for p in ('plan.json','verified_resources_shared.json','control_shared.json','candidate_shared.json','cpu.json','jobs.json')},
        verifier_sha256=sha(Path(__file__)))
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))


if __name__=='__main__':main()
