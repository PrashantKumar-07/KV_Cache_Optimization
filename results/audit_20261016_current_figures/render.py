"""Rebuild current figures from all frozen, verified raw predictions; CPU only.

No product sources, raw evidence, or deployed figures are modified. Historical
plotting modules supply typography, palettes, and the original grouped-bar form.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'results/full_evaluation_20261015'
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
import plot_figures_v2 as style
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from cost_model import CostModel

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof = json.loads((RUN / 'verified_results.json').read_text())
plan = json.loads((RUN / 'plan.json').read_text())
assert proof['verified'] and proof['execution_complete']
assert proof['verified_request_count'] == 10318 and proof['verified_population_rows'] == 1474
assert sha(RUN / 'plan.json') == proof['plan_sha256']
for source in ('src/cost_model.py', 'experiments/longbench_eval.py',
               'experiments/plot_figures_v2.py', 'experiments/plot_publication_figures.py'):
    assert sha(ROOT / source) == plan['source_sha256'][source], source

TASKS = plan['tasks']  # Original figures1/7 use this order.
TRAFFIC_TASKS = ['multifieldqa_en', 'hotpotqa', 'triviaqa', 'qasper', 'narrativeqa', 'gov_report']
METHODS = ['full', 'streamingllm', 'h2o', 'snapkv', 'tieredkv']
METHOD_ARMS = dict(full='full', streamingllm='streamingllm', h2o='h2o_adapted', snapkv='snapkv_adapted')
LABELS = dict(full='Full Cache', streamingllm='Streaming (adapted)', h2o='H2O (adapted)',
              snapkv='SnapKV (adapted)', tieredkv='TieredKV')
BASE_LABELS = style.TASK_LABELS.copy()
style.MLABEL.update(LABELS)
style.TASK_LABELS.update({t: f"{BASE_LABELS[t]}\n(n={plan['counts'][t]})" for t in TASKS})
style.OUT = str(OUT)
plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'current-full-evaluation-20261016'})
cm = CostModel()
cost_prices = {key: vars(value) for key, value in cm.tiers.items()}
frozen_rows = {(r['task'], r['source_zero_based_index']): r for r in proof['rows']}
raw_values = []
endpoint_values = []
record_hashes = {}
stats_keys = ['demoted', 'promoted', 'promotions_deferred', 'paid_writes', 'writes_saved', 'dropped']
max_cost_error = {'modeled_latency_us': 0.0, 'modeled_energy_nj': 0.0}

def near(a, b, tolerance=1e-9):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=tolerance), (a, b)

for relative, digest in sorted(proof['record_sha256'].items()):
    path = ROOT / relative
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == digest, relative
    record_hashes[relative] = digest
    r = json.loads(content)
    t, a, index = r['task'], r['arm'], r['physical_file_row_index']
    verified = frozen_rows[t, index]
    assert r['sample_id'] == verified['sample_id']
    assert r['execution_complete'] and r['plan_sha256'] == proof['plan_sha256']
    near(r['score'], verified['scores'][a])
    stats = r['cache_stats']
    assert stats == verified['resources'][a]['cache_stats']
    assert r['decode_attended_width'] == verified['resources'][a]['decode_attended_width']
    is_tiered = a.startswith('tiered_')
    first = r['geometry_after_prefill_before_first_decode']
    last = r['geometry_after_last_model_call_and_policy_commit']
    layers = len(first['layers'])
    assert layers == 32
    heads, dim = first['layers'][0]['visible_k']['shape'][1::2]
    assert heads == (32 if a == 'snapkv_faithful' else 8) and dim == 128
    assert r['generated_tokens'] - 1 == stats['decode_steps']
    values = dict(task=t, arm=a, source_index=index, sample_id=r['sample_id'],
        score=r['score'], generated_tokens=r['generated_tokens'], decode_steps=stats['decode_steps'],
        modeled_latency_us=stats['modeled_latency_us'], modeled_energy_nj=stats['modeled_energy_nj'],
        **{k: stats[k] if is_tiered else 0 for k in stats_keys})
    # Independently reconstruct the frozen analytical accounting, including the
    # initial exposure but excluding the final exposure never used by a forward.
    width_sum = r['decode_attended_width']['sum']
    cold_sum = 0
    if is_tiered:
        assert stats['paid_writes'] + stats['writes_saved'] == stats['demoted']
        final_cold = [layer['visible_k']['shape'][2] - layer['tier_payloads']['sram_k']['shape'][1]
                      for layer in last['layers']]
        assert len(set(final_cold)) == 1
        cold_sum = (stats['stt_exposed_total'] - final_cold[0]) * layers
        assert 0 <= cold_sum <= width_sum
    hot_sum = width_sum - cold_sum
    elems_per_row = 2 * heads * dim
    attention_lat, attention_eng = cm.gpu_vram_read_cost(hot_sum * elems_per_row)
    attention_lat += cm.read_latency_us('STT-RAM', cold_sum * elems_per_row)
    attention_eng += cm.read_energy_nj('STT-RAM', cold_sum * elems_per_row)
    promotion_lat, promotion_eng = cm.promote_cost(values['promoted'] * elems_per_row)
    demotion_lat, demotion_eng = cm.demote_cost(values['paid_writes'] * elems_per_row)
    drop_lat, drop_eng = cm.drop_cost(values['dropped'] * elems_per_row)
    for key, expected in [('modeled_latency_us', attention_lat + promotion_lat + demotion_lat + drop_lat),
                          ('modeled_energy_nj', attention_eng + promotion_eng + demotion_eng + drop_eng)]:
        error = abs(expected - stats[key])
        max_cost_error[key] = max(error, max_cost_error[key])
        near(expected, stats[key], .051)  # Stored policy stats round each request to0.1.
    values.update(attention_energy_nj_reconstructed=attention_eng,
        promotion_energy_nj=promotion_eng, paid_demotion_energy_nj=demotion_eng,
        drop_energy_nj=drop_eng, attention_energy_nj_residual=stats['modeled_energy_nj'] - promotion_eng - demotion_eng - drop_eng,
        cold_layer_row_reads=cold_sum, hot_layer_row_reads=hot_sum,
        attended_width_min=r['decode_attended_width']['min'], attended_width_max=r['decode_attended_width']['max'])
    assert values['attention_energy_nj_residual'] >= 0
    raw_values.append(values)
    for stage, geometry in [('after_prefill', first), ('after_final_commit', last)]:
        visible = sum(layer[k]['logical_bytes'] for layer in geometry['layers'] for k in ('visible_k', 'visible_v'))
        retained = sum(tensor['logical_bytes'] for layer in geometry['layers']
                       for tensor in layer['tier_payloads'].values()) if is_tiered else visible
        endpoint_values.append(dict(task=t, arm=a, source_index=index, stage=stage,
            retained_logical_bytes=retained, visible_logical_bytes=visible,
            unique_payload_backing_bytes=geometry['unique_payload_backing_bytes']))
assert len(raw_values) == 10318
assert len({(r['task'], r['source_index'], r['arm']) for r in raw_values}) == 10318

def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

write_csv('request_values.csv', raw_values)
write_csv('endpoint_values.csv', endpoint_values)
aggregates = {}
task_values = []
for t in TASKS:
    for a in plan['arms']:
        rows = [r for r in raw_values if r['task'] == t and r['arm'] == a]
        scores = [r['score'] for r in rows]
        assert len(rows) == plan['counts'][t]
        mean, se = statistics.mean(scores), statistics.stdev(scores) / math.sqrt(len(scores))
        near(mean, proof['task_summaries'][t]['full_population_descriptive']['means'][a])
        sums = {key: math.fsum(r[key] for r in rows) for key in rows[0]
                if key not in ('task', 'arm', 'source_index', 'sample_id', 'score', 'attended_width_min', 'attended_width_max')}
        aggregates[t, a] = dict(score=mean, n=len(rows), sample_scores=scores, se=se, cache_stats=sums)
        task_values.append(dict(task=t, arm=a, n=len(rows), score=mean, sample_SE=se, **sums))
write_csv('task_values.csv', task_values)

def dataset(tiered, task_order):
    mapping = {**METHOD_ARMS, 'tieredkv': tiered}
    return {'config': {'tasks': task_order, 'methods': METHODS},
            'results': {t: {m: aggregates[t, a] for m, a in mapping.items()} for t in task_order}}

contract = {'schema': 1, 'population': 1474, 'predictions_verified_against_raw': 10318,
    'counts': plan['counts'], 'source_proof_sha256': sha(RUN / 'verified_results.json'),
    'cost_model_prices': cost_prices, 'max_per_request_cost_reconstruction_error': max_cost_error,
    'rounding_tolerance': 'Recorded policy totals round each request to0.1us/nJ; maximum accepted absolute residual0.051.',
    'figures': {}, 'pending': {'fig4_decode_scaling': 'Fresh original-five-method quiet timing required.',
        'fig8_frac_ablation': 'Fresh fraction sweep required.', 'fig15_write_savings': 'Fresh matching ablation required.'}}
output_paths = []
visual_checks = {}

def save(fig, stem, values=None, caption=None, *, bottom=.18):
    if caption is not None:
        fig.text(.5, .01, caption, ha='center', va='bottom', fontsize=7.2, color='#555555', linespacing=1.25)
        fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.canvas.draw()
    for group in getattr(fig, '_value_label_groups', []):
        for _ in range(30):
            boxes = [text.get_window_extent().expanded(1.10, 1.10) for text in group]
            overlap = next(((i,j) for i in range(len(group)) for j in range(i)
                            if boxes[i].overlaps(boxes[j])), None)
            if overlap is None:
                break
            i,j = overlap
            text = group[i]
            display = text.axes.transData.transform(text.get_position())
            text.set_position(text.axes.transData.inverted().transform(display + [0, 12]))
            fig.canvas.draw()
        assert not any(group[i].get_window_extent().overlaps(group[j].get_window_extent())
                       for i in range(len(group)) for j in range(i))
    # Verify every bar/image value exported by matplotlib and preserve those
    # artists alongside the semantic data; this prevents undocumented plotting
    # transformations from being mistaken for source values.
    artist_values = []
    for ax in fig.axes:
        artist_values.append({'bars': [{'x':float(p.get_x()), 'y':float(p.get_y()), 'width':float(p.get_width()),
            'height':float(p.get_height())} for p in ax.patches if hasattr(p, 'get_height')],
            'images': [np.asarray(im.get_array()).tolist() for im in ax.images],
            'scatter_offsets': [np.asarray(collection.get_offsets()).tolist() for collection in ax.collections
                                if type(collection).__name__ == 'PathCollection'],
            'lines': [{'x':np.asarray(line.get_xdata()).tolist(), 'y':np.asarray(line.get_ydata()).tolist()}
                      for line in ax.lines]})
    contract['figures'][stem] = {'values': values, 'caption': caption, 'artists': artist_values}
    for ext in ('png', 'pdf'):
        target = OUT / f'{stem}.{ext}'
        fig.savefig(target, metadata={'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {})
        output_paths.append(target)
    visual_checks[stem] = {'axes': len(fig.axes), 'artist_values_finite': all(math.isfinite(x['height'])
        for a in artist_values for x in a['bars'])}
    plt.close(fig)

def compressed_labels(t):
    return f"{BASE_LABELS[t]}\n(n={plan['counts'][t]})"

def quality_plot(arm, stem, title):
    data = dataset(arm, TASKS)
    values = {t: {m: {'mean': data['results'][t][m]['score'], 'sample_SE': data['results'][t][m]['se'],
        'n': data['results'][t][m]['n']} for m in METHODS} for t in TASKS}
    caption = ('All 1,474 source records; descriptive, including previously consumed data. Error bars: within-task sample SE (not 95% CI).\n'
        + ('Compressed methods share 1,024 hot-row budget; Tiered adds cold storage/attention; Full is unrestricted.' if arm == 'tiered_hot1024'
           else 'Compressed methods: at most 1,024 cached rows; Tiered 992 hot + up to 32 cold. Extra cold storage; Full is unrestricted.'))
    def callback(fig, actual_stem):
        assert actual_stem == stem
        ax = fig.axes[0]
        expected = [values[t][m]['mean'] for m in METHODS for t in TASKS]
        assert len(ax.patches) == len(expected)
        for p, number in zip(ax.patches, expected): near(p.get_height(), number)
        # Keep the original below-axis five-method legend and rotated labels.
        fig.texts[0].set_text(caption)
        fig.texts[0].set_y(.005)
        fig.set_size_inches(9.6, 4.7)
        fig.tight_layout(rect=(0,.12,1,1))
        save(fig, stem, values, None)
        contract['figures'][stem]['caption'] = caption
    style.savefig = callback
    style.grouped_accuracy(data, stem, title, caption, style='pro')

quality_plot('tiered_hot1024', 'fig1_accuracy_equivram', 'LongBench accuracy: matched 1,024 hot-row budget')
quality_plot('tiered_hot992', 'fig7_accuracy_fair', 'LongBench accuracy: matched cached-attention budget')

# Figure14 preserves its original task order and five-column green heatmap.
matrix = [[aggregates[t, {**METHOD_ARMS, 'tieredkv':'tiered_hot1024'}[m]]['score'] for m in METHODS]
          for t in TRAFFIC_TASKS]
fig, ax = plt.subplots(figsize=(8.5,5.1))
im = ax.imshow(matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(5), [LABELS[m].replace(' (adapted)', '\n(adapted)') for m in METHODS], fontsize=9)
ax.set_yticks(range(6), [compressed_labels(t) for t in TRAFFIC_TASKS], fontsize=9)
for i, row in enumerate(matrix):
    best = np.argmax(row[1:]) + 1
    for j, value in enumerate(row):
        ax.text(j,i,f'{value:.1f}',ha='center',va='center',fontsize=9,
            fontweight='bold' if j==best else 'normal',color='white' if value>=65 and j!=best else '#1d3557' if value>40 else 'black',
            bbox=dict(boxstyle='round,pad=.2',fc='white',ec='none',alpha=.85) if j==best else None)
ax.set_title('LongBench score heatmap: hot 1,024 + cold 2,048 TieredKV',fontweight='bold',pad=10)
fig.colorbar(im,ax=ax,shrink=.9).set_label('Score (F1 / ROUGE-L ×100)')
style.despine(ax)
save(fig,'fig14_heatmap',{'tasks':TRAFFIC_TASKS,'methods':METHODS,'scores':matrix},
    'All 1,474 source records; descriptive. Boxes mark the largest compressed-method point estimate, not significance.\nAdapted baselines; extra Tiered cold capacity/attention. Full cache is unrestricted.',bottom=.13)

# Figure2 retains grouped slate/red bars, a logarithmic cost axis, and task order.
fig,ax=plt.subplots(figsize=(9.6,4.8));x=np.arange(6);bw=.68/5
cost_values={m:[aggregates[t,{**METHOD_ARMS,'tieredkv':'tiered_hot992'}[m]]['cache_stats']['modeled_latency_us']/1000
                for t in TRAFFIC_TASKS] for m in METHODS}
texts=[[] for _ in TRAFFIC_TASKS]
for j,m in enumerate(METHODS):
    xs=x+(j-2)*bw;hs=cost_values[m]
    bars=ax.bar(xs,hs,bw*.92,color=style.FIG1_PALETTE[m],hatch='///' if m=='snapkv' else '',
               edgecolor='white',linewidth=.7,label=LABELS[m],zorder=3)
    for i,(xp,h) in enumerate(zip(xs,hs)):
        text=f'{h/1000:.1f}k' if h>=1000 else f'{h:.0f}'
        texts[i].append(ax.text(xp,h*(1.16 if j%2==0 else 1.45),text,ha='center',va='bottom',fontsize=6.5))
ax.set_yscale('log');all_cost=[v for vs in cost_values.values() for v in vs]
ax.set_ylim(min(all_cost)*.55,max(all_cost)*2.4)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,_: f'{value/1000:g}k' if value>=1000 else f'{value:g}'))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xticks(x,[compressed_labels(t) for t in TRAFFIC_TASKS],fontsize=9)
ax.set_ylabel('Modeled traffic time (ms)')
ax.set_title('Modeled decode traffic cost: matched cached-attention budget',fontsize=10.5,fontweight='bold')
ax.legend(ncol=5,loc='upper center',bbox_to_anchor=(.5,-.20),fontsize=8,frameon=False);style.despine(ax)
fig._value_label_groups = texts
save(fig,'fig2_latency_comparison',{'tasks':TRAFFIC_TASKS,'methods':METHODS,'modeled_time_ms':cost_values},
    'All 1,474 records; task totals with different generated lengths. Analytical traffic time, not measured latency or throughput.\nTiered 992 hot / 2,048 cold / 32 quota; assumed STT 50/12.5 GB/s read/write, VRAM 864 GB/s; prefill/compute/interface omitted.',bottom=.20)

# Figure3 keeps the requested single paid/avoided stack with its data table.
paid=[aggregates[t,'tiered_hot992']['cache_stats']['paid_writes'] for t in TRAFFIC_TASKS]
avoided=[aggregates[t,'tiered_hot992']['cache_stats']['writes_saved'] for t in TRAFFIC_TASKS]
totals=np.array(paid)+avoided;percent=100*np.array(avoided)/totals
fig,ax=plt.subplots(figsize=(9.6,5.6));x=np.arange(6)
ax.bar(x,paid,.55,label='Paid writes',color='#c0392b',alpha=.88,zorder=3)
ax.bar(x,avoided,.55,bottom=paid,label='Writes avoided',color='#27ae60',alpha=.88,zorder=3)
for i,(total,pc) in enumerate(zip(totals,percent)):
    ax.text(i,total+max(totals)*.025,f'{pc:.1f}% avoided',ha='center',va='bottom',fontsize=9,color='#1e8449',fontweight='bold')
ax.set_ylim(0,max(totals)*1.16);ax.set_xticks(x,[compressed_labels(t) for t in TRAFFIC_TASKS],rotation=20,ha='right')
ax.set_ylabel('Logical per-layer token write events')
ax.set_title('STT-RAM write traffic: paid versus avoided',fontweight='bold',pad=10)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_:f'{v/1e6:g}M' if v>=1e6 else f'{v/1000:g}k' if v>=1000 else str(int(v))))
ax.legend(loc='upper left');style.despine(ax)
table=ax.table(cellText=[[f'{p:,.0f}' for p in paid],[f'{a:,.0f}' for a in avoided],[f'{r:.1f}%' for r in percent]],
    rowLabels=['Paid','Avoided','Avoided %'],colLabels=[BASE_LABELS[t] for t in TRAFFIC_TASKS],loc='bottom',bbox=[0,-.63,1,.30])
table.auto_set_font_size(False);table.set_fontsize(7)
for cell in table.get_celld().values():cell.set_edgecolor('#cccccc');cell.set_linewidth(.5);cell.PAD=.03
fig.text(.5,.012,'All 1,474 records; Tiered 992 hot / 2,048 cold, initial cold fraction 0.75. Decode demotions summed across 32 layers.\nBackup reuse avoids a payload write; prompt placement/metadata excluded. Simulated events, not measured device endurance.',ha='center',fontsize=7.2,color='#555555')
fig.tight_layout();fig.subplots_adjust(bottom=.43)
save(fig,'fig3_write_savings',{'tasks':TRAFFIC_TASKS,'arm':'tiered_hot992','paid_events':paid,'avoided_events':avoided,'avoided_percent':percent.tolist()})

# Figure6: counters are policy actions; deferrals and shadow reuse are not transfers.
fig,ax=plt.subplots(figsize=(8.5,4.8));bottom=np.zeros(6);migration={}
for key,label,color,alpha,hatch in [('demoted','Logical demotions','#e07a5f',1,''),
    ('promoted','Promoted rows','#457b9d',1,''),('promotions_deferred','Deferred promotion candidates','#8e6bbf',.85,''),
    ('dropped','Cold rows dropped','#c0392b',.7,'///')]:
    values=[aggregates[t,'tiered_hot1024']['cache_stats'][key] for t in TRAFFIC_TASKS];migration[key]=values
    ax.bar(x,values,.55,bottom=bottom,label=label,color=color,alpha=alpha,hatch=hatch,zorder=3);bottom+=values
for i,total in enumerate(bottom):ax.text(i,total*1.015,f'{total/1e6:.1f}M',ha='center',va='bottom',fontsize=7.5)
ax.set_ylim(0,max(bottom)*1.15);ax.set_xticks(x,[compressed_labels(t) for t in TRAFFIC_TASKS],rotation=20,ha='right',fontsize=9)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,_:f'{value/1e6:g}M'))
ax.set_ylabel('Per-layer policy events (sum)')
ax.set_title('TieredKV policy actions by task: hot 1,024 configuration',fontsize=10.5,fontweight='bold')
ax.legend(loc='upper left',fontsize=8);style.despine(ax)
save(fig,'fig6_migration_profile',{'tasks':TRAFFIC_TASKS,'arm':'tiered_hot1024','events':migration},
    'All 1,474 records; decode counters sum 32 layers; generated lengths differ. These stacks are event counts, not bytes.\nDemotions include backup reuse; deferred candidates perform no transfer. Hysteresis and live-cold-floor gates can defer.',bottom=.16)

# Figure11 uses only configurations actually evaluated. The old hypothetical
# equal-total-memory arm is deliberately absent rather than assigned fake data.
arms_groups=[['streamingllm','h2o_adapted','snapkv_adapted'],['tiered_hot1024'],['tiered_hot992']]
capacity_rows=[1024,1024+2048,992+2048]
bytes_per_model_row=32*8*128*2*2
footprint=[]
for arms,cap in zip(arms_groups,capacity_rows):
    endpoints=[r for r in endpoint_values if r['arm'] in arms]
    observed={key:max(endpoints,key=lambda r:r[key]) for key in ('retained_logical_bytes','unique_payload_backing_bytes')}
    footprint.append({'arms':arms,'configured_payload_bytes':cap*bytes_per_model_row,
        **{f'max_endpoint_{key}':r[key] for key,r in observed.items()},
        'witnesses':{key:{k:r[k] for k in ('task','arm','source_index','stage')} for key,r in observed.items()}})
fig,ax=plt.subplots(figsize=(9.5,4.9));xp=np.arange(3)
configured=[r['configured_payload_bytes']/1e6 for r in footprint]
retained=[r['max_endpoint_retained_logical_bytes']/1e6 for r in footprint]
backing=[r['max_endpoint_unique_payload_backing_bytes']/1e6 for r in footprint]
ax.bar(xp,configured,.65,color=['#3b8bc0','#c0392b','#c0392b'],label='Configured hot + cold payload capacity',alpha=.9)
ax.scatter(xp-.1,retained,marker='D',color='#222222',s=30,label='Maximum observed retained payload',zorder=5)
ax.scatter(xp+.1,backing,marker='x',color='#1d3557',s=45,label='Maximum observed payload backing storage',zorder=5)
for i,(c,r,b) in enumerate(zip(configured,retained,backing)):
    ax.text(i,b+max(backing)*.025,f'{c:.1f} MB capacity\n{b:.1f} MB endpoint storage',ha='center',va='bottom',fontsize=8)
ax.set_xticks(xp,['Adapted compressed\n1,024 cached rows','TieredKV hot 1,024\n+ up to 32 exposed cold','TieredKV hot 992\n+ up to 32 exposed cold'])
ax.set_ylim(0,max(backing)*1.36);ax.set_ylabel('KV payload / storage (MB)')
ax.set_title('Configured capacity and observed KV payload storage',fontsize=11,fontweight='bold')
ax.legend(loc='upper left',fontsize=8);style.despine(ax)
save(fig,'fig11_footprint',{'groups':footprint,'bytes_per_configured_row':bytes_per_model_row},
    'All 1,474 records; observed maxima over post-prefill and final commits only, not runtime peaks. Ordinary GPU mode; 1 MB = 10⁶ bytes.\nBacking storage includes visible copies; metadata/weights/activations/allocator reserve omitted. Hypothetical equal-total arm was not evaluated.',bottom=.17)

# Figure12 preserves five-method stacked analytical energy, with exact price/hash
# provenance and per-request independent accounting checks above (no clamp).
energy={};fig,ax=plt.subplots(figsize=(9.5,5.0));xp=np.arange(5);bottom=np.zeros(5)
energy_keys=['attention_energy_nj_residual','promotion_energy_nj','paid_demotion_energy_nj','drop_energy_nj']
for m in METHODS:
    a={**METHOD_ARMS,'tieredkv':'tiered_hot1024'}[m]
    energy[m]={key:math.fsum(aggregates[t,a]['cache_stats'][key] for t in TASKS)/1e9 for key in energy_keys}
for key,label,color in zip(energy_keys,['Attention reads\n(frozen model)','Promote\n(STT read + VRAM write)',
    'Paid demotion\n(VRAM read + STT write)','Drop\n(zero payload cost)'],['#457b9d','#e9c46a','#c0392b','#7f7f7f']):
    values=[energy[m][key] for m in METHODS]
    ax.bar(xp,values,.6,bottom=bottom,label=label,color=color,edgecolor='white',linewidth=.8,zorder=3)
    for i,(base,value) in enumerate(zip(bottom,values)):
        if value>max(sum(v.values()) for v in energy.values())*.025:
            ax.text(i,base+value/2,f'{value:.1f} J',ha='center',va='center',fontsize=7.5,color='white',fontweight='bold')
    bottom+=values
for i,value in enumerate(bottom):ax.text(i,value*1.015,f'{value:.1f} J total',ha='center',va='bottom',fontsize=8)
ax.set_ylim(0,max(bottom)*1.14);ax.set_xticks(xp,[LABELS[m] for m in METHODS],fontsize=9)
ax.set_ylabel('Modeled traffic energy (J)')
ax.set_title('Decode KV-traffic energy at frozen assumed prices',fontweight='bold',pad=10)
ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=8);style.despine(ax)
save(fig,'fig12_energy',{'methods':METHODS,'tiered_arm':'tiered_hot1024','components_J':energy},
    'All 1,474 records; summed free-generation work differs. Assumed STT 2/8 pJ per bit (read/write), VRAM 12 pJ per bit.\nPayload-traffic model only: prefill, compute, interface, metadata and system power omitted. No device energy measurement.',bottom=.22)

# Separate faithful comparator; never substitutes for any five-method main plot.
supp_arms=['full','snapkv_faithful','tiered_hot1024','tiered_hot992']
supp_labels=['Full Cache','Faithful SnapKV 1,024','TieredKV hot 1,024','TieredKV hot 992']
supp_colors=['#2F3B4C','#8e6bbf','#B03A2E','#d97c72']
fig,ax=plt.subplots(figsize=(9.6,5.0));xp=np.arange(6);bw=.68/4;supp={}
for j,(a,label,color) in enumerate(zip(supp_arms,supp_labels,supp_colors)):
    hs=[aggregates[t,a]['score'] for t in TASKS];es=[aggregates[t,a]['se'] for t in TASKS]
    supp[a]={'mean':hs,'sample_SE':es}
    xs=xp+(j-1.5)*bw
    ax.bar(xs,hs,bw*.92,yerr=es,color=color,edgecolor='white',linewidth=.7,label=label,
        error_kw={'capsize':2,'elinewidth':.7,'ecolor':'#333333'},zorder=3)
    for xval,h,e in zip(xs,hs,es):ax.text(xval,h+e+1,f'{h:.1f}',ha='center',va='bottom',rotation=90,fontsize=7,fontweight='bold')
ax.set_ylim(0,max(v+e for a in supp for v,e in zip(supp[a]['mean'],supp[a]['sample_SE']))*1.19)
ax.set_xticks(xp,[compressed_labels(t) for t in TASKS]);ax.set_ylabel('Score (F1 / ROUGE-L ×100)')
ax.set_title('Supplement: faithful SnapKV quality comparison',fontweight='bold')
ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.20),fontsize=8,frameon=False);style.despine(ax)
primary=proof['primary_contrast']
save(fig,'supplement_faithful_accuracy',{'tasks':TASKS,'arms':supp_arms,'scores':supp,'primary':primary},
    'Full 1,474-record descriptive means; error bars=sample SE. Faithful Snap expands KV heads and grows during decode; unequal resources.\n'
    f"Preregistered five-task previously unconsumed stratum: Tiered hot 992 − faithful Snap = {primary['point_estimate']:+.2f} points; 95% cluster CI [{primary['ci95'][0]:.2f},{primary['ci95'][1]:.2f}]. No superiority.",bottom=.22)

(OUT/'numeric_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
(OUT/'raw_record_sha256.json').write_text(json.dumps(record_hashes,indent=2)+'\n')
(OUT/'visual_validation.json').write_text(json.dumps({'machine_checks':visual_checks,'manual_visual_review':'pending'},indent=2)+'\n')
inputs=[RUN/'verified_results.json',RUN/'plan.json',RUN/'partitions.json',ROOT/'src/cost_model.py',
    ROOT/'experiments/plot_figures_v2.py',ROOT/'experiments/plot_publication_figures.py',ROOT/'experiments/longbench_eval.py']
manifest={'status':'staged for root visual review; no deployed figures changed','generator_sha256':sha(__file__),
    'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
    'all10318_raw_record_hashes_rechecked':True,'raw_hash_manifest_sha256':sha(OUT/'raw_record_sha256.json'),
    'data_csv_sha256':{name:sha(OUT/name) for name in ('request_values.csv','endpoint_values.csv','task_values.csv')},
    'numeric_contract_sha256':sha(OUT/'numeric_contract.json'),
    'outputs_sha256':{p.name:sha(p) for p in output_paths},
    'scientific_scope':'Complete original-source1474-row descriptive evaluation; consumed data disclosed; main plots preserve original five-method questions and per-figure task order. Faithful baseline and failed primary contrast are supplemental. No latency/SOTA/whole-memory/device-energy inference.',
    'pending':contract['pending'],'deployment_performed':False}
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'staged_outputs':len(output_paths),'raw_records_verified':len(raw_values),
    'max_cost_rounding_error':max_cost_error,'pending':contract['pending']},indent=2))
