"""Extend verified decode counters with actual initial cold KV placement.

No new generation, model tuning, or physical endurance inference. The
counterfactual holds the measured trace fixed and disables backup write reuse;
it is an accounting bound, not a rerun of an exclusive-cache policy.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,importlib.util,json

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=BASE/'whole_request_writes.json';assert not out.exists()
    source=ROOT/'results/full_evaluation_20261015/verified_results.json'
    proof=json.loads(source.read_text());assert proof['verified'] and proof['execution_complete']
    helper=ROOT/'results/audit_20261016_headroom/analyze_whole_writes.py'
    spec=importlib.util.spec_from_file_location('write_accounting',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sums=defaultdict(lambda:[0,0,0]);rows=[];hashes={}
    expected={(r['task'],r['source_zero_based_index']):r for r in proof['rows']}
    for name,digest in proof['record_sha256'].items():
        if not name.endswith('_tiered_hot1024.json'):continue
        p=ROOT/name;assert sha(p)==digest
        r=json.loads(p.read_text());assert r['execution_complete'] and r['arm']=='tiered_hot1024'
        key=(r['task'],r['physical_file_row_index']);assert r['sample_id']==expected[key]['sample_id']
        assert r['score']==expected[key]['scores']['tiered_hot1024']
        initial=0
        layers=r['geometry_after_prefill_before_first_decode']['layers'];assert len(layers)==32
        for layer in layers:
            tensors=layer['tier_payloads'];k=tensors['stt_k'];v=tensors['stt_v'];shape=k['shape']
            assert shape[0]==8 and shape[2]==128 and v['shape']==shape
            assert k['dtype']==v['dtype']=='torch.bfloat16'
            assert k['logical_bytes']==v['logical_bytes']==shape[1]*8*128*2
            initial+=shape[1]
        stats=r['cache_stats'];paid=stats['paid_writes'];saved=stats['writes_saved']
        assert paid+saved==stats['demoted']
        assert all(isinstance(x,int) and x>=0 for x in (initial,paid,saved))
        for population in ('all_tasks',r['task']):
            for i,x in enumerate((initial,paid,saved)):sums[population][i]+=x
        rows.append({'task':r['task'],'index':key[1],'sample_id':r['sample_id'],
            'generated_tokens':r['generated_tokens'],**module.counters(initial,paid,saved)})
        hashes[name]=digest
    assert len(rows)==len(expected)==1474
    assert len({(r['task'],r['index']) for r in rows})==1474
    report={'verified':True,'script_sha256':sha(Path(__file__)),
        'reused_counter_definition_sha256':sha(helper),'quality_proof_sha256':sha(source),
        'records':1474,'raw_record_sha256':hashes,'rows':rows,
        'summaries':{key:module.counters(*value) for key,value in sums.items()},
        'scope':'Logical cold KV payload placement: initial retained cold K/V once plus paid decode demotions. Each row is one layer-token with8KVheads/D128/BF16 K+V =4096bytes. Fixed-trace counterfactual writes every demotion; no causal exclusive-policy rerun. Variable generation lengths remain measured. Initial GPU prefill, metadata, physical write amplification, device energy and endurance are excluded. Do not infer physical lifetime gains.',
        'initial_write_source':'src/tiered_kv_cache.py initial_bifurcation: cold K/V clone, not charged by decode StepRecord.',
        'figure_action':'Keep original figure3 decode-only question/style. Whole-request numbers must accompany any broader write or endurance discussion; no changed figure values fabricated.'}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summaries'],indent=2))

if __name__=='__main__':main()
