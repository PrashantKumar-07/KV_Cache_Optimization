"""Independent source/replay/gate audit; failed screen stops model expansion."""
import csv,hashlib,json,tarfile
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_stage.json';assert not output.exists()
    raw=read(BASE/'stage.json');plan=read(BASE/'plan.json');frozen=read(BASE/'stage_implementation.json')
    assert sha(BASE/'plan.json')==raw['plan_sha256']==frozen['plan_sha256']
    assert sha(ROOT/plan['diagnostic'])==plan['diagnostic_sha256']
    assert sha(BASE/'control_source.tar.gz')==plan['control_archive_sha256']
    with tarfile.open(BASE/'control_source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==plan['source_sha256'] and all(sha(ROOT/f)==h for f,h in sources.items())
    assert sha(BASE/'stage_source.tar.gz')==frozen['archive_sha256']
    with tarfile.open(BASE/'stage_source.tar.gz') as t:
        sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert sources==frozen['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    for name in ['tests_job.json','stage_job.json']:
        job=read(BASE/name);assert job['status']=='finished' and job['exit_code']==0
    assert '19 passed' in (BASE/'tests_focused.log').read_text() and '347 passed' in (BASE/'tests_full.log').read_text()
    assert raw['passed'] and len(raw['checks'])==6
    assert {r['n_new'] for r in raw['checks']}=={1,4,32}
    assert all(r.get('victims_exact') or r.get('victims_and_projection_bits_exact') for r in raw['checks'])
    assert len(raw['quiet_observations'])==25 and all(r['quiet'] and not r['unexpected_compute_processes'] for r in raw['quiet_observations'])
    assert len(raw['rows'])==12
    steps={arm:[] for arm in ['control','candidate']};blocks=[]
    for block in range(3):
        rows=raw['rows'][block*4:block*4+4];assert [r['arm'] for r in rows]==['control','candidate','candidate','control']
        for row in rows:
            assert row['block']==block and len(row['warmup_ms'])==4 and len(row['step_ms'])==20
            assert all(v>0 and np.isfinite(v) for v in row['step_ms'])
            steps[row['arm']].extend(row['step_ms'])
        blocks.append({arm:float(np.median([v for r in rows if r['arm']==arm for v in r['step_ms']])) for arm in steps})
    summary={arm:dict(median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95))) for arm,v in steps.items()}
    assert summary==raw['summary'] and blocks==raw['blocks']
    saved=summary['control']['median_ms']-summary['candidate']['median_ms']
    guards=dict(saved_ms=saved>=4,all_blocks_faster=all(r['candidate']<r['control'] for r in blocks),p95=summary['candidate']['p95_ms']<=1.02*summary['control']['p95_ms'])
    assert guards==raw['guards'] and saved==raw['median_saving_ms']
    passing=all(guards.values());assert passing==raw['charged_stage_primary_passed']
    assert not passing
    result=dict(verified=True,tests_passed=347,focused_tests_passed=19,gpu_exact_stage_cases=6,quiet_gpu0_observations=25,
        complete_preparation_selection_summary=summary,median_saved_ms=saved,p95_ratio=summary['candidate']['p95_ms']/summary['control']['p95_ms'],guards=guards,
        charged_stage_primary_passed=passing,fullmodel_expansion=False,ordinary_source_cli_defaults_changed=False,ordinary_adoption=False,full_goal_complete=False,
        all_owned_jobs_terminal=True,current_source_archives_verified=True,
        disposition='Stop projected-victim candidate: <4ms saved and p95>2%worse. No fullmodel/parity/resource/timing/quality expansion; 347tests are not proof of fullmodel candidate preservation.',
        next='Target ordinary layer-state transitions/synchronization, not this failed projection-only stage. Preregister before edits, charge complete mechanism and compare fullmodel/frozen states before adoption.',
        limits=raw['limits']+' GPU process checks are before/after only; no favorable timing rerun.',
        evidence_sha256={f:sha(BASE/f) for f in ['plan.json','control_source.tar.gz','stage_implementation.json','stage_source.tar.gz','stage.json','stage.log','stage_job.json','tests_focused.log','tests_full.log','tests_job.json','check_stage.py','verify_stage.py']})
    with (BASE/'stage_summary.csv').open('w',newline='') as h:
        writer=csv.writer(h);writer.writerow(['scope','metric','control','candidate'])
        for metric in ['median_ms','p95_ms']:writer.writerow(['synthetic_complete_projection_selection',metric,summary['control'][metric],summary['candidate'][metric]])
    result['stage_csv_sha256']=sha(BASE/'stage_summary.csv')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['verified','tests_passed','charged_stage_primary_passed','median_saved_ms','p95_ratio','fullmodel_expansion','full_goal_complete']},indent=2))
if __name__=='__main__':main()
