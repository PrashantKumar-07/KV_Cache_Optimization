"""Read-only evidence checks for research native paired current-query selection."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name):return json.loads((BASE/name).read_text())
def archive(name,sources):
    with tarfile.open(BASE/name) as t:
        for path,value in sources.items():
            assert hashlib.sha256(t.extractfile(path).read()).hexdigest()==value,path

def main():
    plan=read('plan.json');v1=read('implementation.json');v2=read('implementation_v2.json');fix=read('correction_plan.json')
    for record,file,sources in [(plan,'control_source.tar.gz',plan['source_sha256']),
            (v1,'implementation_source.tar.gz',v1['sources']),(v2,'implementation_v2_source.tar.gz',v2['sources'])]:
        assert sha(BASE/file)==record['archive_sha256'];archive(file,sources)
    assert sha(BASE/'plan.json')==v1['plan_sha256']
    assert sha(BASE/'implementation.json')==fix['original_implementation_sha256']==v2['original_implementation_sha256']
    assert sha(BASE/'correction_plan.json')==v2['correction_plan_sha256']
    assert sha(BASE/'gpu.json')==fix['failed_raw_sha256']
    assert sha(BASE/'gpu_failed_job.json')==fix['failed_job_sha256']
    for path,value in plan['source_sha256'].items():assert sha(ROOT/path)==value,path
    for path,value in v2['sources'].items():assert sha(ROOT/path)==value,path
    changes=sorted(path for path in v1['sources'] if v1['sources'][path]!=v2['sources'][path])
    assert changes==sorted(v2['changes'])
    with tarfile.open(BASE/'implementation_source.tar.gz') as t:
        path='experiments/pool_query_selector.py';old=t.extractfile(path).read().decode()
        assert (ROOT/path).read_text().replace('torch.where(~cache.stt_shadow)[0].to(cache.cfg.device)',
            'torch.where(~cache.stt_shadow)[0]')==old
        path=str((BASE/'check_gpu.py').relative_to(ROOT));old=t.extractfile(path).read().decode()
        assert (ROOT/path).read_text().replace("output=base/'gpu_v2.json'","output=base/'gpu.json'")==old
    for record,field,file in [(v1,'tests_focused_sha256','tests_focused.log'),
            (v1,'tests_full_sha256','tests_full.log'),(v2,'tests_full_sha256','tests_full_v2.log')]:
        assert sha(BASE/file)==record[field]
        assert ('4 passed' if 'focused' in file else '266 passed') in (BASE/file).read_text()
    failed=read('gpu.json');good=read('gpu_v2.json')
    assert not failed['passed'] and 'indices should be either on cpu' in failed['error']
    assert read('gpu_failed_job.json')['exit_code']==1
    assert read('gpu_v2_job.json')==dict(session_id=24710,status='finished',exit_code=0)
    assert good['passed'] and good['hooks_canonical_method_restored']
    assert (good['context'],good['decode'])==(4096,12)
    arms=good['arms'];assert [a['arm'] for a in arms]==['gpu_original','gpu_paired','cpu_paired']
    for a in arms:
        assert len(a['predicted_token_ids'])==12 and len(a['frames'])==13 and len(a['selections'])==384
        for field in ('predicted_token_ids','frames','selections'):assert a[field]==arms[0][field]
        assert a['canonical_reads_during_selector']==0
    for a in arms[1:]:assert a['selected_layers']==384 and a['scanned_rows']>0
    assert arms[2]['host_read_bytes']>0
    assert "args.cpu_batched_scores and args.strategy != 'lru'" in (ROOT/'experiments/evaluate_query_selection.py').read_text()
    names=['plan.json','control_source.tar.gz','implementation.json','implementation_source.tar.gz',
        'implementation_v2.json','implementation_v2_source.tar.gz','correction_plan.json','gpu.json',
        'gpu_v2.json','gpu_failed_job.json','gpu_v2_job.json','gpu.log','gpu_v2.log',
        'tests_focused.log','tests_full.log','tests_full_v2.log','verify_correctness.py']
    report=dict(passed=True,decision='Correctness passed; proceed only to fresh16K50 resource gate.',
        exact_prediction_steps=12,exact_cache_frames=13,exact_visible_kv_layer_forwards=384,
        native_query_canonical_reads=0,native_cpu_h2d_bytes=arms[2]['host_read_bytes'],
        ordinary_source_and_cli_unchanged=True,tests_passed=266,ordinary_adoption=False,
        full_goal_complete=False,limits='Single model/seed4K12 correctness. No latency, independent quality, '
            'actual baseline resource equality, full512MiB, or physicalSTT proof.',
        evidence_sha256={name:sha(BASE/name) for name in names})
    output=BASE/'verified_correctness.json';assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
