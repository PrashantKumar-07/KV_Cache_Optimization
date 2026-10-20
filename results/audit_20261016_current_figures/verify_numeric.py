"""Independent raw-record -> plotted-artist audit; never imports the renderer.

Uses two-pass sample variance, direct recorded counters, physical byte formulas,
and raw endpoint tensors. No GPU, plotting modules, or mutable live CostModel.
"""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'results/full_evaluation_20261015'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof = json.loads((RUN/'verified_results.json').read_text())
plan = json.loads((RUN/'plan.json').read_text())
contract = json.loads((HERE/'numeric_contract.json').read_text())
manifest = json.loads((HERE/'provenance.json').read_text())
assert proof['verified'] and proof['execution_complete']
assert contract['source_proof_sha256'] == sha(RUN/'verified_results.json')
assert manifest['numeric_contract_sha256'] == sha(HERE/'numeric_contract.json')
assert manifest['generator_sha256'] == sha(HERE/'render.py')
for name,digest in manifest['input_sha256'].items(): assert sha(ROOT/name) == digest
for name,digest in manifest['outputs_sha256'].items(): assert sha(HERE/name) == digest
for name,digest in manifest['data_csv_sha256'].items(): assert sha(HERE/name) == digest
records = {}
by_task_arm = defaultdict(list)
endpoint = defaultdict(list)
for filename,digest in proof['record_sha256'].items():
    payload = (ROOT/filename).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == digest
    r = json.loads(payload)
    key = r['task'],r['physical_file_row_index'],r['arm']
    assert key not in records
    records[key] = r
    by_task_arm[r['task'],r['arm']].append(r)
    for stage,key_stage in [('after_prefill','geometry_after_prefill_before_first_decode'),
                            ('after_final_commit','geometry_after_last_model_call_and_policy_commit')]:
        geometry = r[key_stage]
        retained = 0
        for layer in geometry['layers']:
            tensors = layer.get('tier_payloads', {'k':layer['visible_k'],'v':layer['visible_v']})
            for tensor in tensors.values():
                expected = math.prod(tensor['shape']) * 2
                assert tensor['dtype'] == 'torch.bfloat16' and tensor['logical_bytes'] == expected
                retained += expected
        endpoint[r['arm']].append({'task':r['task'],'arm':r['arm'],'source_index':r['physical_file_row_index'],
            'stage':stage,'retained_logical_bytes':retained,
            'unique_payload_backing_bytes':geometry['unique_payload_backing_bytes']})
assert len(records) == 10318

def equal(left,right):
    if isinstance(left,(list,tuple)):
        assert len(left) == len(right), (len(left),len(right))
        for a,b in zip(left,right): equal(a,b)
    elif isinstance(left,dict):
        assert set(left)==set(right),(set(left),set(right))
        for k in left:equal(left[k],right[k])
    elif isinstance(left,(int,float)):
        assert math.isclose(left,right,rel_tol=1e-11,abs_tol=1e-8),(left,right)
    else:assert left==right,(left,right)

def mean_se(task,arm):
    values=[r['score'] for r in by_task_arm[task,arm]]
    n=len(values);mean=sum(values)/n
    variance=sum((v-mean)**2 for v in values)/(n-1)
    return mean,math.sqrt(variance/n),n

def total(task,arm,key):
    return sum(r['cache_stats'][key] for r in by_task_arm[task,arm])

tasks=['multifieldqa_en','qasper','narrativeqa','hotpotqa','triviaqa','gov_report']
traffic=['multifieldqa_en','hotpotqa','triviaqa','qasper','narrativeqa','gov_report']
methods=['full','streamingllm','h2o','snapkv','tieredkv']
armmap=dict(full='full',streamingllm='streamingllm',h2o='h2o_adapted',snapkv='snapkv_adapted')
figs=contract['figures'];checks={}
for name,tiered in [('fig1_accuracy_equivram','tiered_hot1024'),('fig7_accuracy_fair','tiered_hot992')]:
    f=figs[name];mapping={**armmap,'tieredkv':tiered}
    for t in tasks:
        for m in methods:
            mean,se,n=mean_se(t,mapping[m]);equal(f['values'][t][m],{'mean':mean,'sample_SE':se,'n':n})
    equal([b['height'] for b in f['artists'][0]['bars']],[mean_se(t,mapping[m])[0] for m in methods for t in tasks])
    lines=f['artists'][0]['lines'];assert len(lines)==10
    for i,m in enumerate(methods):
        equal(lines[2*i]['y'],[mean_se(t,mapping[m])[0]-mean_se(t,mapping[m])[1] for t in tasks])
        equal(lines[2*i+1]['y'],[mean_se(t,mapping[m])[0]+mean_se(t,mapping[m])[1] for t in tasks])
    checks[name]='30 means, sample SE/n, bar heights, and60 rendered errorbar endpoints match raw scores'

f=figs['fig14_heatmap'];expected=[[mean_se(t,{**armmap,'tieredkv':'tiered_hot1024'}[m])[0] for m in methods] for t in traffic]
equal(f['values']['tasks'],traffic);equal(f['values']['methods'],methods);equal(f['values']['scores'],expected)
equal(f['artists'][0]['images'][0],expected);checks['fig14_heatmap']='30 raw means match image matrix and task/method order'

f=figs['fig2_latency_comparison'];expected={m:[total(t,{**armmap,'tieredkv':'tiered_hot992'}[m],'modeled_latency_us')/1000 for t in traffic] for m in methods}
equal(f['values']['modeled_time_ms'],expected)
equal([b['height'] for b in f['artists'][0]['bars']],[v for m in methods for v in expected[m]])
checks['fig2_latency_comparison']='30 time sums from raw request counters; exact microseconds-to-milliseconds conversion and bars'

f=figs['fig3_write_savings'];paid=[total(t,'tiered_hot992','paid_writes') for t in traffic];avoided=[total(t,'tiered_hot992','writes_saved') for t in traffic]
equal(f['values']['paid_events'],paid);equal(f['values']['avoided_events'],avoided)
equal(f['values']['avoided_percent'],[100*b/(a+b) for a,b in zip(paid,avoided)])
bars=f['artists'][0]['bars'];equal([b['height'] for b in bars],paid+avoided);equal([b['y'] for b in bars],[0]*6+paid)
for t in traffic: equal(total(t,'tiered_hot992','demoted'),total(t,'tiered_hot992','paid_writes')+total(t,'tiered_hot992','writes_saved'))
checks['fig3_write_savings']='12 raw summed write counters,6 ratios, stack heights/bottoms and demotion accounting'

f=figs['fig6_migration_profile'];keys=['demoted','promoted','promotions_deferred','dropped'];expected={k:[total(t,'tiered_hot1024',k) for t in traffic] for k in keys}
equal(f['values']['events'],expected);bars=f['artists'][0]['bars'];bottom=[0]*6
for i,k in enumerate(keys):
    equal([b['height'] for b in bars[i*6:(i+1)*6]],expected[k]);equal([b['y'] for b in bars[i*6:(i+1)*6]],bottom)
    bottom=[a+b for a,b in zip(bottom,expected[k])]
checks['fig6_migration_profile']='24 raw policy action sums, stack heights/bottoms; deferrals remain nontransfer counters'

f=figs['fig11_footprint'];expected_groups=[(['streamingllm','h2o_adapted','snapkv_adapted'],1024),(['tiered_hot1024'],3072),(['tiered_hot992'],3040)]
retained=[];backing=[];configured=[]
for entry,(arms,rows) in zip(f['values']['groups'],expected_groups):
    equal(entry['arms'],arms);capacity=rows*32*8*128*2*2;equal(entry['configured_payload_bytes'],capacity)
    candidates=[r for arm in arms for r in endpoint[arm]]
    for metric in ['retained_logical_bytes','unique_payload_backing_bytes']:
        maximum=max(r[metric] for r in candidates);equal(entry['max_endpoint_'+metric],maximum)
        witness=entry['witnesses'][metric]
        found=[r for r in candidates if all(r[k]==v for k,v in witness.items())];assert len(found)==1;equal(found[0][metric],maximum)
    configured.append(capacity/1e6);retained.append(entry['max_endpoint_retained_logical_bytes']/1e6);backing.append(entry['max_endpoint_unique_payload_backing_bytes']/1e6)
equal([b['height'] for b in f['artists'][0]['bars']],configured)
scatter=f['artists'][0]['scatter_offsets'];assert len(scatter)==2
equal([v[1] for v in scatter[0]],retained);equal([v[1] for v in scatter[1]],backing)
checks['fig11_footprint']='3 frozen configured capacities and6 maxima/markers independently checked over20636 raw endpoints, including witnesses'

# Independent SI formula, no imported live cost model: one8-head K/V row is
# 4096B, promotion14pJ/bit, paid demotion20pJ/bit, drop0 payload cost.
prices=contract['cost_model_prices'];assert prices['STT-RAM']['read_pj_per_bit']==2 and prices['STT-RAM']['write_pj_per_bit']==8
assert prices['GPU_VRAM']['read_pj_per_bit']==prices['GPU_VRAM']['write_pj_per_bit']==12
f=figs['fig12_energy'];energy={}
for m in methods:
    a={**armmap,'tieredkv':'tiered_hot1024'}[m]
    tot=sum(total(t,a,'modeled_energy_nj') for t in tasks)/1e9
    p=sum(total(t,a,'promoted') for t in tasks)*4096*8*14/1e12 if m=='tieredkv' else 0
    d=sum(total(t,a,'paid_writes') for t in tasks)*4096*8*20/1e12 if m=='tieredkv' else 0
    energy[m]={'attention_energy_nj_residual':tot-p-d,'promotion_energy_nj':p,'paid_demotion_energy_nj':d,'drop_energy_nj':0}
equal(f['values']['components_J'],energy)
keys=list(energy['full']);bottom=[0]*5
for i,k in enumerate(keys):
    bars=f['artists'][0]['bars'][i*5:(i+1)*5];equal([b['height'] for b in bars],[energy[m][k] for m in methods]);equal([b['y'] for b in bars],bottom)
    bottom=[v+energy[m][k] for v,m in zip(bottom,methods)]
checks['fig12_energy']='20 energy components and stack geometry from raw totals/counters and direct frozen SI energy prices'

f=figs['supplement_faithful_accuracy'];arms=['full','snapkv_faithful','tiered_hot1024','tiered_hot992']
for a in arms:
    equal(f['values']['scores'][a]['mean'],[mean_se(t,a)[0] for t in tasks]);equal(f['values']['scores'][a]['sample_SE'],[mean_se(t,a)[1] for t in tasks])
equal([b['height'] for b in f['artists'][0]['bars']],[mean_se(t,a)[0] for a in arms for t in tasks])
for i,a in enumerate(arms):
    equal(f['artists'][0]['lines'][2*i]['y'],[mean_se(t,a)[0]-mean_se(t,a)[1] for t in tasks])
    equal(f['artists'][0]['lines'][2*i+1]['y'],[mean_se(t,a)[0]+mean_se(t,a)[1] for t in tasks])
equal(f['values']['primary'],proof['primary_contrast'])
partition=json.loads((RUN/'partitions.json').read_text())
diffs=defaultdict(list)
for r in partition['rows']:
    if r['partition']=='unconsumed_as_of_inventory':
        t,index=r['task'],r['source_zero_based_index']
        diffs[t].append(records[t,index,'tiered_hot992']['score']-records[t,index,'snapkv_faithful']['score'])
assert len(diffs)==5 and sum(map(len,diffs.values()))==739
equal(sum(sum(v)/len(v) for v in diffs.values())/5,proof['primary_contrast']['point_estimate'])
checks['supplement_faithful_accuracy']='24 raw means/SE, bars/errorbars; primary point from739 raw paired rows and CI bound to independent verified proof (not bootstrapped again)'

with (HERE/'task_values.csv').open() as stream:
    for row in csv.DictReader(stream):
        t,a=row['task'],row['arm'];mean,se,n=mean_se(t,a)
        equal(float(row['score']),mean);equal(float(row['sample_SE']),se);equal(int(row['n']),n)
assert not any((HERE/(name+'.png')).exists() for name in contract['pending'])
assert len(manifest['outputs_sha256'])==18
result={'verified':True,'GPU_used':False,'raw_records_rehashed':len(records),'raw_endpoint_snapshots_checked':len(records)*2,
    'checks':checks,'data_CSV_and_output_hashes_verified':True,'main_five_methods_preserved':True,
    'per_figure_original_task_order_preserved':True,'confidence_semantics':'Main/supplement bars use within-task sample SE; only frozen primary annotation uses its verified cluster95% interval.',
    'pending_figures_not_fabricated':contract['pending'],'verifier_sha256':sha(__file__),
    'numeric_contract_sha256':sha(HERE/'numeric_contract.json'),'provenance_sha256':sha(HERE/'provenance.json')}
(HERE/'numeric_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'verified':True,'figures':len(checks),'raw_records':len(records)},indent=2))
