"""Final read-only audit of current source, terminal jobs and status artifacts."""
import hashlib,json,tarfile
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def main():
    output=BASE/'verified_status.json';assert not output.exists()
    names=['verified_correctness.json','verified_resource_v2.json','verified_cost.json','verified_lru.json']
    checked=0
    for name in names:
        proof=read(BASE/name);assert proof['verified'] and not proof['ordinary_adoption'] and not proof['full_goal_complete']
        for relative,expected in proof['evidence_sha256'].items():
            assert sha(BASE/relative)==expected,(name,relative);checked+=1
    for prefix,archive_name in [('', 'implementation_source.tar.gz'),('resource_','resource_source.tar.gz'),
                                ('cost_','cost_source.tar.gz'),('lru_','lru_source.tar.gz')]:
        metadata=read(BASE/(prefix+'implementation.json'))
        assert sha(BASE/(prefix+'plan.json'))==metadata['plan_sha256']
        assert sha(BASE/archive_name)==metadata['archive_sha256']
        with tarfile.open(BASE/archive_name) as archive:
            archived={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                      for m in archive.getmembers() if m.isfile()}
        assert archived==metadata['sources']
        assert all(sha(ROOT/relative)==expected for relative,expected in archived.items())
    plan=read(BASE/'plan.json')
    assert all(sha(ROOT/relative)==expected for relative,expected in plan['source_sha256'].items())
    native=read(BASE/'resource_native_source.json')
    assert sha(BASE/'resource_native_source.tar.gz')==native['archive_sha256']
    assert all(sha(ROOT/relative)==expected for relative,expected in native['sources'].items())
    jobs={name:read(BASE/(name+'_job.json')) for name in ['gpu','resource','cost','lru','tests']}
    assert all(job['status']=='finished' and job['exit_code']==0 for job in jobs.values())
    correctness=read(BASE/'verified_correctness.json');assert correctness['tests_passed']==306
    resource=read(BASE/'verified_resource_v2.json')
    assert resource['resource_primary_passed'] and not resource['full512mib_passed']
    cost=read(BASE/'verified_cost.json');assert cost['gpu_visible_suffix_component_primary_passed']
    guard=read(BASE/'verified_lru.json');assert not guard['ordinary_lru_regression_guards_passed']
    assert guard['fullscan_quality_timing_expansion'] is False
    manifest=read(BASE/'figure_manifest.json')
    assert manifest['component_proof_sha256']==sha(BASE/'verified_cost.json')
    assert manifest['lru_proof_sha256']==sha(BASE/'verified_lru.json')
    assert manifest['plot_source_sha256']==sha(BASE/'plot_results.py')
    assert all(sha(BASE/name)==expected for name,expected in manifest['artifacts_sha256'].items())
    docs=['PROJECT_STATUS.md','START_HERE.md','RESEARCH_ROADMAP.md','PAPER_WORKING_DRAFT.md',
          'REMEDIATION_CHECKLIST.md','SUBMISSION_SPRINT.md']
    for name in docs:
        current=(ROOT/name).read_text()[:3000]
        for text in ['306','545.17','7.85%','1.686','1.559','Stop standalone','10–20','terminal']:
            assert text in current,(name,text)
        assert 'nowrunning' not in current and 'is running' not in current
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),verified=True,
        classification='Progress: preserved visible-buffer ownership and exact query behavior; component cost passed; ordinaryLRU guard failed.',
        tests_passed=306,evidence_hash_checks=checked,source_archives_equal_current=True,
        ordinary_source_cli_defaults_unchanged=True,jobs=jobs,all_owned_jobs_terminal=True,
        figure_visually_inspected_no_overlap=True,resource_native_source_supplement_matches_measured_and_frozen_fingerprints=True,
        resource_active_mib=resource['candidate_active_decode_envelope_mib'],
        component_median_gain_percent=100*cost['relative_median_improvement'],
        ordinary_lru_median_ratio=guard['ratios']['median_decode'],ordinary_lru_p95_ratio=guard['ratios']['p95_decode'],
        ordinary_adoption=False,fullscan_quality_timing_expansion=False,full_goal_complete=False,
        next='Preregister a distinct lower-traffic reader: keys for unchanged ranking, values only for selected rows. Require parity/ownership/resource/traffic/component/LRU gates. No stopped fullscan quality retry.',
        remaining_working_days_estimate=[10,20],estimate_condition='Conditional on hardware and research outcomes.',
        evidence_sha256={name:sha(BASE/name) for name in names+['figure_manifest.json','README.md','verify_status.py']},
        status_sha256={name:sha(ROOT/name) for name in docs+['EXPERIMENT_LOG.md']})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['verified','tests_passed','evidence_hash_checks','component_median_gain_percent',
        'ordinary_lru_median_ratio','all_owned_jobs_terminal','full_goal_complete']},indent=2))
if __name__=='__main__':main()
