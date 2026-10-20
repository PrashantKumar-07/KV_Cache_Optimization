"""Read-only final standalone append stop audit; original failed logs retained."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    output=BASE/'verified_status.json';assert not output.exists()
    names=['verified_correctness.json','verified_resource.json'];checked=0
    for name in names:
        proof=read(BASE/name);assert proof['verified'] and not proof['ordinary_adoption'] and not proof['full_goal_complete']
        for f,h in proof['evidence_sha256'].items():assert sha(BASE/f)==h,(name,f);checked+=1
    for stage in ['gpu','resource']:
        frozen=read(BASE/(stage+'_implementation.json'))
        assert sha(BASE/(stage+'_source.tar.gz'))==frozen['archive_sha256']
        with tarfile.open(BASE/(stage+'_source.tar.gz')) as t:sources={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
        assert sources==frozen['sources'] and all(sha(ROOT/f)==h for f,h in sources.items())
    jobs={p.name:read(p) for p in BASE.glob('*job.json')};assert all(j['status']=='finished' for j in jobs.values())
    resource=read(BASE/'verified_resource.json');assert not resource['resource_primary_passed'] and not resource['ready_for_wholeloop_cost_gate']
    assert not (BASE/'cost.json').exists()
    result=dict(verified=True,tests_passed=374,evidence_hash_checks=checked,jobs=jobs,all_owned_jobs_terminal=True,source_archives_equal_current=True,candidate_active_mib=resource['candidate_active_mib'],resource_primary_passed=False,standalone_expansion_stopped=True,ordinary_adoption=False,full_goal_complete=False,limits='Initial failed CPU fixture logs retained without original execution source snapshots. This audit does not assert unrelated later GPU jobs are terminal.',evidence_sha256={f:sha(BASE/f) for f in names+['README.md','verify_status.py']})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['verified','candidate_active_mib','all_owned_jobs_terminal','full_goal_complete']}))
if __name__=='__main__':main()
