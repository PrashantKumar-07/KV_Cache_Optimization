"""Verify weak logits lifetime without changing frozen model execution."""
import hashlib
import json
from pathlib import Path
import tarfile
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=json.loads((BASE/'plan.json').read_text())
    assert sha(BASE/'source.tar.gz')==plan['source_bundle_sha256']
    assert sha(BASE.parent/'source.tar.gz')==plan['primary_bundle_sha256']
    hashes={}
    for archive_path in (BASE.parent/'source.tar.gz',BASE/'source.tar.gz'):
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                if member.isfile():hashes[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    assert hashes['experiments/audit_prefill_logits_lifetime.py']==plan['driver_sha256']
    evidence={}
    for arm in ('fused_bounded','native_reuse'):
        raw=json.loads((BASE/f'{arm}.json').read_text())
        original=json.loads((BASE.parent/f'shared_0_{arm}.json').read_text())
        assert raw['passed'] and raw['actual_arm']==arm
        assert raw['logits_audit_source_sha256']==plan['driver_sha256']
        assert all(hashes[p]==value for p,value in raw['actual_source_sha256'].items())
        for field in ('state_sha256','predicted_token_ids','counters','occupied_host_bytes','reserved_host_bytes',
            'prompt_sha256','teacher_source_sha256','teacher_token_ids','input_ids_sha256',
            'host_block_binary_sha256','gpu_phase_peaks','host_block_observations'):
            assert raw[field]==original[field],(arm,field)
        rows=raw['logits_lifetime_rows'];assert len(rows)==51
        assert rows[0]['logits_shape']==[1,16384,32000] and rows[0]['logits_dtype']=='torch.bfloat16'
        assert rows[0]['logits_storage_bytes']==1*16384*32000*2==1048576000
        assert not rows[0]['prefill_logits_alive_before_forward'] and rows[0]['prefill_logits_alive_after_forward']
        assert rows[1]['prefill_logits_alive_before_forward'] and rows[1]['prefill_logits_alive_after_forward']
        for i,row in enumerate(rows[1:],start=2):
            assert row['call']==i and row['logits_shape']==[1,1,32000] and row['logits_storage_bytes']==64000
            if i>=3:assert not row['prefill_logits_alive_before_forward'] and not row['prefill_logits_alive_after_forward']
        evidence[f'{arm}.json']=sha(BASE/f'{arm}.json')
    result=dict(verified=True,fresh_processes=2,prefill_logits_storage_mib=1000,
        alive_through_first_decode_forward=True,released_before_second_decode_forward=True,
        no_extra_payload_owners=True,model_outputs_counters_and_gpu_peaks_unchanged=True,
        evidence_sha256=evidence,source_bundle_sha256=sha(BASE/'source.tar.gz'),verifier_sha256=sha(Path(__file__)),
        limitations='Untimed weak C++ storage observation under the measured fresh-process harness. Legacy profile scopes prefill output inside a function and does not keep these logits through decode. Confirms an avoidable live owner in this harness, not a demonstrated optimization or complete explanation of every temporary/peak. No model argument/default changed.')
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
