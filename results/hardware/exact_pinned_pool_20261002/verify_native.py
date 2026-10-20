"""Verify exact allocation/owner and native request-preservation screens."""
import hashlib
import json
from pathlib import Path
import tarfile
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=json.loads((BASE/'systems_plan.json').read_text())
    assert sha(BASE/'source.tar.gz')==plan['source_bundle_sha256']
    assert sha(BASE/'allocation.json')==plan['allocation_proof_sha256']
    manifest={}
    with tarfile.open(BASE/'source.tar.gz') as archive:
        for member in archive.getmembers():
            if member.isfile():manifest[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    assert manifest==json.loads((BASE/'source_manifest.json').read_text())
    allocation=json.loads((BASE/'allocation.json').read_text());assert allocation['passed'] and len(allocation['checks'])==19
    assert all(r['exact'] for r in allocation['checks'])
    assert [r['bytes'] for r in allocation['checks'][:6]]==[1,31,32,33,151257088,276824064]
    assert [r['consumer'] for r in allocation['checks'][6:18]]==[0,1]*6
    assert allocation['checks'][-1]['case']=='delayed_d2h_last_view_owner'
    def balances(events):
        live={}
        for event in events:
            if event['kind']=='allocate':assert event['token'] not in live;live[event['token']]=event['bytes']
            else:assert live.pop(event['token'])==event['bytes']
        assert not live
    balances(allocation['allocation_events'])
    native=json.loads((BASE/'native_multi_pool.json').read_text())
    assert native['passed'] and native['exact_pinned_root'] and len(native['checks'])==4
    assert sum(r['exact_read_observations'] for r in native['checks'])==768
    assert all(r['queued_write_resets']==1 for r in native['checks'])
    evidence={}
    for name,expected_cases,capacity in [('native_multi_pool',4,None),('boundary_shared',36,1154),('boundary_default',4,2112)]:
        raw=json.loads((BASE/(name+'.json')).read_text());assert raw['passed'] and raw['exact_pinned_root']
        assert all(manifest[p]==value for p,value in raw['exact_pool_source_sha256'].items())
        assert len(raw['checks'])==expected_cases and raw['exact_allocation_stats']['live_allocations']==0
        assert raw['exact_allocation_stats']['live_requested_bytes']==0
        balances(raw['exact_allocation_stats']['events'])
        if capacity:
            assert max(r['transient_logical_peak'] for r in raw['checks'])==capacity
            assert all(r['decode_calls']==24 and r['exact_state_payload_counters'] and r['rejected_append_before_mutation'] for r in raw['checks'])
            assert raw['native_reuse_launches']['calls']>0 and sum(r['eager_compactions'] for r in raw['checks'])>0
        evidence[name+'.json']=sha(BASE/(name+'.json'))
    assert '40 passed' in (BASE/'factory_suite.txt').read_text()
    result=dict(verified=True,allocation_and_owner_cases=19,native_read_observations=768,boundary_cases=40,
        live_allocations_after_native_resets=0,factory_and_nearest_suite_tests=40,advance_to_model_memory_and_regression=True,
        evidence_sha256=evidence|{'allocation.json':sha(BASE/'allocation.json')},source_bundle_sha256=sha(BASE/'source.tar.gz'),verifier_sha256=sha(Path(__file__)),
        limitations='Exact public CUDA host ranges, same-context delayed owner release and unchanged native mapped transfer paths. Existing cache math/capacities preserve; model/ABBA/generation/recycling/fresh-memory/CLI gates still mandatory. No model-wide memory, speed, serving-thread safety or physical STT claim.')
    path=BASE/'verified_native.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
