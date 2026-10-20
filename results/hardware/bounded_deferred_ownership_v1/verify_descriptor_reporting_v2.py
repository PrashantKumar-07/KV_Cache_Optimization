"""Reporting-only correction; original proof retained, no benchmark retry."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_descriptor_reporting_v2.json';assert not output.exists()
    cost=read(BASE/'verified_cost.json');raw=read(BASE/'cost.json');frozen=read(BASE/'cost_implementation.json')
    assert cost['verified'] and not cost['bounded_deferred_wholeloop_primary_passed']
    assert cost['candidate_extra_metadata_transfers']==0
    assert sha(BASE/'cost.json')==cost['evidence_sha256']['cost.json']
    assert sha(ROOT/'experiments/native_reuse_visible.py')==frozen['sources']['experiments/native_reuse_visible.py']
    source=(ROOT/'experiments/native_reuse_visible.py').read_text()
    for term in ['np.int64','np.int32',"'sram_k','sram_v','sram_cum','sram_age'",'key.data_ptr(),value.data_ptr()','state.sram_k.shape[1],key.shape[2]']:assert term in source
    for row in raw['rows']:
        if row['arm']=='query':assert row['shared_transition_counts']['bounded_native_groups']==200
    # Six int64 pointers, two int32 sizes and 991 int32 indices per layer,
    # two calls per group, 8 layers/group, 200 measured groups/trial.
    per_layer=6*8+2*4+991*4
    bytes_per_trial=per_layer*8*2*200;assert bytes_per_trial==12864000
    result=dict(verified=True,correction='Original verified_cost.json scalar candidate_extra_metadata_transfers=0 is incorrect. Native KV manager payload/call replay remains exact; native gather descriptors are separate Torch CPU-to-GPU uploads.',candidate_descriptor_h2d_bytes_per_measured_trial_derived=bytes_per_trial,candidate_descriptor_upload_calls_per_measured_trial_derived=200*2*3,derivation='Six int64 pointers + two int32 sizes + 991 int32 keep indices per layer; 8 layers/group, 2 passes/group, 200 groups/trial. Three .to(device) calls per native helper. Derived from fixed geometry/source, not independent PCIe transaction measurement.',total_extra_metadata_vs_ordinary='Not independently instrumented in this run; do not assert zero or equate descriptor bytes to physical bus transactions.',wholeloop_timing_charges_descriptor_work=True,resource_peaks_charge_descriptor_allocations=True,raw_timings_and_failed_decision_unchanged=True,benchmark_rerun=False,ordinary_adoption=False,full_goal_complete=False,evidence_sha256={f:sha(BASE/f) for f in ['verified_cost.json','cost.json','cost_implementation.json','verify_descriptor_reporting_v2.py']},native_wrapper_sha256=sha(ROOT/'experiments/native_reuse_visible.py'))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['verified','candidate_descriptor_h2d_bytes_per_measured_trial_derived','candidate_descriptor_upload_calls_per_measured_trial_derived','benchmark_rerun']}))
if __name__=='__main__':main()
