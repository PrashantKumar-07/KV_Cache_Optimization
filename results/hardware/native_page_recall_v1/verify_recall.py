"""Frozen native-query stable8-page recall screen and exact trajectory replay."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    plan=read(BASE/'plan.json');frozen=read(BASE/'implementation.json');raw=read(BASE/'recall.json')
    assert sha(BASE/'plan.json')==frozen['plan_sha256']==raw['plan_sha256']
    assert sha(BASE/'control_source.tar.gz')==plan['archive_sha256']
    assert sha(BASE/'implementation_source.tar.gz')==frozen['archive_sha256']
    for file,sources in [('control_source.tar.gz',plan['source_sha256']),('implementation_source.tar.gz',frozen['sources'])]:
        with tarfile.open(BASE/file) as t:
            archived={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
        assert archived==sources and all(sha(ROOT/f)==v for f,v in sources.items())
    assert sha(ROOT/plan['reference'])==plan['reference_sha256']
    job=read(BASE/'job.json');assert job['status']=='finished' and job['exit_code']==0
    assert raw['passed'] and raw['own_hooks_canonical_method_restored']
    ref=read(ROOT/plan['reference']);result=raw['result'];rows=raw['records']
    assert len(rows)==1600 and len({(r['layer'],r['step']) for r in rows})==1600
    assert len(result['frames'])==51 and len(result['selections'])==1600
    assert result['selected_layers']==1600 and result['canonical_reads_during_selector']==0
    assert result['predicted_token_ids']==ref['predicted_token_ids']
    assert result['frames'][-1]['state_sha256']==ref['state_sha256'] and result['frames'][-1]['counters']==ref['counters']
    for row in rows:
        assert row['quota']==32 and row['live_rows']>32 and row['candidate_rows']>=32
        assert row['pages_scanned']>=min(8,row['active_pages'])
        assert 0<=row['oracle_overlap']<=32 and 0<row['mass_ratio']<=1+1e-6
        assert not row['underfilled'] and row['diagnostic_native_h2d_bytes']==row['live_rows']*8*128*2*2
    recall=float(np.mean([r['oracle_overlap']/32 for r in rows]))
    mass=float(np.mean([r['mass_ratio'] for r in rows]))
    fraction=float(np.mean([r['candidate_rows']/r['live_rows'] for r in rows]))
    guards=dict(mean_top32_overlap=recall>=.90,mean_full_oracle_mass_ceiling=mass>=.95,
        mean_candidate_fraction=fraction<=.25,zero_underfill=not any(r['underfilled'] for r in rows))
    passing=all(guards.values());diagnostic_bytes=sum(r['diagnostic_native_h2d_bytes'] for r in rows)
    assert result['host_read_bytes']==ref['prototype_counts']['host_read_bytes']+diagnostic_bytes
    # Historical claims remain scoped to their old GPU-cold implementation.
    history={}
    for name,key in [('stable_page_report.json','stable_page_report_sha256'),('indexed_report.json','indexed_report_sha256')]:
        path=ROOT/'results/page_recall'/name;assert sha(path)==plan['history'][key]
        doc=read(path)
        for source,value in doc.get('inputs_sha256',doc.get('input_sha256',{})).items():assert sha(path.parent/source)==value
        history[name]=doc
    summary_by_layer={}
    for layer in range(32):
        layerrows=[r for r in rows if r['layer']==layer]
        assert len(layerrows)==50
        summary_by_layer[str(layer)]=dict(mean_top32_overlap=float(np.mean([r['oracle_overlap']/32 for r in layerrows])),
            mean_full_oracle_mass_ceiling=float(np.mean([r['mass_ratio'] for r in layerrows])))
    result_doc=dict(verified=True,diagnostic_screen_passed=passing,guards=guards,records=1600,
        mean_top32_overlap=recall,mean_full_oracle_mass_ceiling=mass,mean_candidate_fraction=fraction,
        mean_candidate_rows=float(np.mean([r['candidate_rows'] for r in rows])),
        mean_live_cold_rows=float(np.mean([r['live_rows'] for r in rows])),
        minimum_layer_mean_top32_overlap=min(v['mean_top32_overlap'] for v in summary_by_layer.values()),
        p05_mass_ceiling=float(np.quantile([r['mass_ratio'] for r in rows],.05)),
        max_extension_pages=max(r['pages_scanned'] for r in rows),
        projected_paired_candidate_payload_bytes=sum(r['candidate_rows']*8*128*2*2 for r in rows),
        full_key_diagnostic_native_h2d_bytes=diagnostic_bytes,
        observed_native_h2d_bytes_including_diagnostic=raw['result']['host_read_bytes'],
        max_layer_live_centroid_summary_bytes=max(r['centroid_summary_bytes'] for r in rows),
        exact_native_query_predictions_state_counters_replayed=True,own_hooks_canonical_guard_restored=True,
        all_owned_jobs_finished=True,ordinary_source_changed=False,tests_passed=266,
        fixed8page_implementation_expansion_allowed=passing,ordinary_adoption=False,full_goal_complete=False,
        decision='Diagnostic permits implementation research; independent quality/timing/resource gates still required.' if passing else
            'Stop implementation/quality/timing of this fixed8-page centroid heuristic: engineering recall screen failed.',
        per_layer=summary_by_layer,
        historical_gpu_cold_indexed_itl_change_pct=history['indexed_report.json']['indexed_itl_change_pct'],
        metric_definition='Candidates chosen by max-head centroid page score. Selected mass is the BEST top32 under '
            'the full-key reference distribution within those candidates, an optimistic ceiling. Candidate-only softmax '
            'may change head denominators/ranking; this diagnostic does not establish local-rerank or generation quality.',
        limits='One model/seed16K50 full native query teacher trajectory. Summaries rebuilt from full keys offline, '
            'not maintained live. Projected candidate payload excludes index maintenance/query ranking/additional transfers; '
            'it is not a measured H2D saving or timing claim. Full512MiB, actual faithful-baseline matching, independent NI '
            'and physicalSTT remain open.',
        evidence_sha256={f:sha(BASE/f) for f in ('plan.json','control_source.tar.gz','implementation.json','implementation_source.tar.gz',
            'recall.json','job.json','verify_recall.py')})
    output=BASE/'verified_report.json';assert not output.exists()
    output.write_text(json.dumps(result_doc,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result_doc.items() if k not in ('per_layer','evidence_sha256')},indent=2))
if __name__=='__main__':main()
