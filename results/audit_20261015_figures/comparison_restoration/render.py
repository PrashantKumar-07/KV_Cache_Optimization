"""Restore original scientific questions/design; do not substitute 3/2-arm pilots.

Original five-method data are archived evaluation results, not current claims.
Figure3 contains only paid/avoided writes, without the unrelated budget sweep.
"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import plot_figures_v2 as v2
import plot_publication_figures as pub
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs={}
def load(name):
 p=ROOT/name;inputs[name]=sha(p);return json.loads(p.read_text())
v2.OUT=str(OUT)
ev=load('results/remediation_v3/equal_vram_merged.json')
fair=load('results/remediation_v3/equal_attended_merged.json')
v2.grouped_accuracy(ev,'fig1_accuracy_equivram','LongBench accuracy: equal logical VRAM',
 'Archived evaluation; adapted baselines, extra Tiered cold capacity. Error bars = sample SE. Current-source reruns pending.',style='pro')
v2.grouped_accuracy(fair,'fig7_accuracy_fair','LongBench accuracy: matched attended cache budget',
 'Archived evaluation; adapted baselines. Tiered: 992 hot + ≤32 cold; baselines: 1024. Error bars = sample SE.',style='pro')
save=v2.savefig
def heatmap_save(fig,stem):
 fig.text(.5,.01,'Archived evaluation; adapted baselines. Boxes mark point estimates, not significance. Current-source reruns pending.',ha='center',fontsize=8,color='#555555')
 fig.tight_layout(rect=(0,.05,1,1));save(fig,stem)
v2.savefig=heatmap_save;v2.fig14_heatmap(ev);v2.savefig=save
# Preserve the other original questions as well: modeled task traffic and
# measured five-method context scaling remain separate plots.
modeled=load('results/final_equal_attended_frac075.json')
old_pub_save=pub.savefig
def modeled_save(fig,outdir,stem):
 fig.axes[0].set_title('Modeled decode traffic cost: matched attended cache budget',fontsize=10.5,fontweight='bold')
 fig.axes[0].set_ylabel('Summed modeled decode traffic time (ms)')
 fig.texts[0].set_text('Archived evaluation; adapted baselines, unequal generated lengths. Analytical totals, not measured or per-token speed.')
 for ext in ['png','pdf']:fig.savefig(OUT/f'{stem}.{ext}')
 plt.close(fig)
pub.savefig=modeled_save;pub.fig2_latency(modeled,str(OUT));pub.savefig=old_pub_save
lat=load('results/remediation_v3/latency.json')
def scaling_save(fig,stem):
 fig.axes[0].set_title('Decode latency including attention feedback and cache policy')
 fig.texts[0].set_text('Archived L40S evaluation; all five original methods. Same real prompt prefix, 50 decode steps; adapted baselines.')
 save(fig,stem)
v2.savefig=scaling_save;v2.fig4_decode_scaling(lat);v2.savefig=save
# Retain original red/green stacked-bar + per-task table visual grammar.
data=load('results/final_equal_attended_frac075.json');tasks=data['config']['tasks']
paid=[];saved=[]
for task in tasks:
 c=data['results'][task]['tieredkv']['cache_stats'];paid.append(c['paid_writes']);saved.append(c['writes_saved'])
paid=np.array(paid);saved=np.array(saved);total=paid+saved
assert np.all(total>0)
pct=100*saved/total;x=np.arange(len(tasks));labels=[pub.TASK_LABELS.get(t,t) for t in tasks]
fig,ax=plt.subplots(figsize=(9.6,5.2));bw=.55
ax.bar(x,paid,bw,label='Paid writes',color='#c0392b',alpha=.88,zorder=3)
ax.bar(x,saved,bw,bottom=paid,label='Writes avoided',color='#27ae60',alpha=.88,zorder=3)
for i,(p,s,r) in enumerate(zip(paid,saved,pct)):
 ax.text(i,p+s+total.max()*.025,f'{r:.1f}% avoided',ha='center',va='bottom',fontsize=9,color='#1e8449',fontweight='bold')
ax.set_ylim(0,total.max()*1.15);ax.set_xticks(x,labels,rotation=25,ha='right')
ax.set_ylabel('Logical per-layer token write events')
ax.set_title('STT-RAM write traffic: paid versus avoided',fontweight='bold',pad=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f'{v/1000:.0f}k' if v>=1000 else str(int(v))))
ax.legend(loc='upper left');v2.despine(ax)
table=ax.table(cellText=[[f'{p/1000:.0f}k' for p in paid],[f'{s/1000:.0f}k' for s in saved],[f'{r:.1f}%' for r in pct]],
 rowLabels=['Paid','Avoided','Avoided %'],colLabels=labels,loc='bottom',bbox=[0,-.52,1,.31])
table.auto_set_font_size(False);table.set_fontsize(8)
for cell in table.get_celld().values():cell.set_edgecolor('#cccccc');cell.set_linewidth(.5);cell.PAD=.03
fig.text(.5,.012,'Archived fraction 0.75 evaluation; decoder demotions only. No budget-sweep panel.\nCounters exclude prompt placement and metadata; these are simulated payload events, not physical endurance measurements.',ha='center',fontsize=8,color='#555555')
fig.tight_layout();fig.subplots_adjust(bottom=.4)
for ext in ['png','pdf']:fig.savefig(OUT/f'fig3_write_savings.{ext}')
plt.close(fig)
import plot_frac_ablation as frac
paths=sorted((ROOT/'results').glob('e2_frac*_short.json'))
for p in paths:inputs[str(p.relative_to(ROOT))]=sha(p)
rows=sorted(frac.load_frac(str(p)) for p in paths)
fig,ax=plt.subplots(figsize=(7.5,4.2));fracs=[r[0] for r in rows]
ax.plot(fracs,[r[1] for r in rows],'o-',color='#1d3557',label='Avg F1 (5 tasks)')
ax.set_xlabel('STT-RAM bifurcation fraction (headroom $\\leftarrow$)')
ax.set_ylabel('Avg F1 ($\\times$ 100)',color='#1d3557');ax.tick_params(axis='y',labelcolor='#1d3557');ax.invert_xaxis()
other=ax.twinx();other.plot(fracs,[r[2] for r in rows],'s--',color='#27ae60',label='Write savings %')
other.set_ylabel('Write savings (%)',color='#27ae60');other.tick_params(axis='y',labelcolor='#27ae60');other.set_ylim(0,100)
ax.set_title('STT-RAM headroom: accuracy–write-savings trade-off',fontsize=10.5,fontweight='bold')
for f,s,v in rows:ax.annotate(f'{s:.1f}',(f,s),textcoords='offset points',xytext=(0,8),ha='center',fontsize=8)
fig.text(.5,.005,'Archived five-task evaluation; decode-demotion ratios. Point estimates do not establish accuracy non-inferiority.',ha='center',fontsize=7,color='#555555')
fig.tight_layout(rect=(0,.04,1,1))
for ext in ['png','pdf']:fig.savefig(OUT/f'fig8_frac_ablation.{ext}')
plt.close(fig)
report={'request':'Restore original5method/6task figure structure; preservequestions andstyle; Fig3singlewritepanel only',
 'inputs_sha256':inputs,'generator_sha256':sha(Path(__file__)),
 'outputs_sha256':{p.name:sha(p) for p in OUT.glob('fig*') if p.suffix in ['.png','.pdf']},
 'figure3_values':{'tasks':tasks,'paid':paid.tolist(),'avoided':saved.tolist(),'avoided_pct':pct.tolist()},
 'figure1_7_14_data_status':'Actual archived evaluations with recorded provenance, not currentsource/superiority evidence; completecurrent5methodcomparisons needadditionalruns.',
 'separate_development_figures':'Prior3/2armdevelopmentcharts preserved elsewhere; not substituted fororiginalquestions'}
(OUT/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
