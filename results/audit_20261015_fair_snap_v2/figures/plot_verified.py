"""CPU-only current measured figures, preserving shared publication rc/palette.

Reads only the independently verified terminal proof. Never overwrites originals.
"""
from pathlib import Path
import csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;BASE=OUT.parent
sys.path.insert(0,str(ROOT/'experiments'))
import plot_figures_v2 as style
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proofpath=BASE/'independently_verified_benchmark.json';p=json.loads(proofpath.read_text())
assert p['independently_verified'] is True
rawpath=BASE/'benchmark.json';assert sha(rawpath)==p['evidence_sha256'][str(rawpath.relative_to(ROOT))]
contexts=sorted(map(int,p['per_context']));arms=['snap','native'];labels={'snap':'Faithful SnapKV','native':'TieredKV native'}
colors={'snap':style.PALETTE['snapkv'],'native':style.PALETTE['tieredkv']}
plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'tieredkv-fair-benchmark-v2'})
rows=[]
for ctx in contexts:
 for arm in arms:
  d=p['per_context'][str(ctx)]['arms'][arm]
  rows.append({'context_tokens':ctx,'arm':arm,**d})
with (OUT/'timing_values.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
terms=[]
for arm in arms:
 candidates=p['scoped_memory_terms']['16384'][arm];selected=max(enumerate(candidates),key=lambda v:v[1]['scoped_bytes'])
 for name,value in selected[1]['terms_bytes'].items():terms.append(dict(context_tokens=16384,arm=arm,measured_trial_index=selected[0],term=name,bytes=value,MiB=value/2**20))
with (OUT/'memory_values.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(terms[0]));w.writeheader();w.writerows(terms)
outputs=[];captions={};artists={}

def save(fig,name,caption,values):
 captions[name]=caption;artists[name]=values
 fig.text(.5,.015,caption,ha='center',va='bottom',fontsize=8,color='#555555',linespacing=1.3)
 fig.tight_layout(rect=(0,.17,1,1))
 for ext in ('pdf','png'):
  target=OUT/f'{name}.{ext}';fig.savefig(target,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {});outputs.append(target)
 plt.close(fig)

def status(metric):
 return 'PASS' if all(p['per_context'][str(c)]['guards'][metric] for c in contexts) and p['pooled']['guards'][metric] else 'FAIL'

fig,ax=plt.subplots(figsize=(9.5,4.6));x=np.arange(len(contexts));width=.32;figvalues={}
for i,arm in enumerate(arms):
 values=[p['per_context'][str(c)]['arms'][arm]['median_ttft_ms'] for c in contexts];figvalues[arm]=values
 bars=ax.bar(x+(i-.5)*width,values,width*.94,color=colors[arm],edgecolor='white',linewidth=.6,label=labels[arm],zorder=3)
 ax.bar_label(bars,labels=[f'{v:,.0f}' for v in values],padding=4,fontsize=8)
ax.set_xticks(x,[f'{c:,}' for c in contexts]);ax.set_xlabel('Context length (tokens)');ax.set_ylabel('Median steady-request TTFT (ms)')
ax.set_title('Measured time to first token — faithful SnapKV vs native TieredKV',fontweight='bold',pad=10)
ax.set_ylim(0,max(v for values in figvalues.values() for v in values)*1.23);ax.legend(loc='upper left');style.despine(ax)
save(fig,'fig2_latency_comparison',f'Preloaded Mistral-7B / L40S; 3 ABBA blocks, 12 measured requests per arm/context. TTFT includes required policy work.\nNative/Snap TTFT ≤1.10 guard: {status("median_ttft_ms")}; setup/reset excluded; diagnostic/startup observations excluded.\nBoth configurations have a 512 MiB scoped budget; not exact whole-memory equality. CPU cold storage; STT costs are simulated.',figvalues)

fig,ax=plt.subplots(figsize=(9.5,4.6));figvalues={}
for arm in arms:
 for metric,linestyle,marker,statname in [('median_itl_ms','-','o','median'),('p95_itl_ms','--','s','p95')]:
  values=[p['per_context'][str(c)]['arms'][arm][metric] for c in contexts];figvalues[arm+'_'+metric]=values
  ax.plot(contexts,values,linestyle=linestyle,marker=marker,color=colors[arm],label=labels[arm]+' '+statname,zorder=3)
ax.set_xscale('log');ax.set_xticks(contexts,[f'{c:,}' for c in contexts]);ax.minorticks_off();ax.set_xlabel('Context length (tokens)');ax.set_ylabel('Full-loop decode latency (ms/token)')
ax.set_title('Measured decode latency including attention feedback and cache policy',fontweight='bold',pad=10);ax.legend(ncol=2,loc='best');style.despine(ax)
ax.set_ylim(bottom=0)
save(fig,'fig4_decode_scaling',f'600 measured forwards per arm/context; 50 fixed teacher-forced steps/request. Solid: median; dashed: p95, not uncertainty.\nNative/Snap median ≤0.95: {status("median_itl_ms")}; p95 ≤1.05: {status("p95_itl_ms")}. One text, three correlated prefixes.\nBoth configurations have a 512 MiB scoped budget; not exact whole-memory equality. CPU cold storage; STT costs are simulated.',figvalues)

fig,ax=plt.subplots(figsize=(9.5,5.5));positions=np.arange(2);bottom=np.zeros(2)
keys=[('decode_GPU_excluding_static_model','GPU allocated decode peak − static model',None,''),('observed_main_pinned_allocator_block','Exact pinned CPU root (also held idle)','#3b8bc0',''),('canonical_CPU_arrays','Canonical CPU arrays','#2a7f62',''),('CPU_streak_array_peak','CPU streak scratch peak','#e07a5f','///'),('CPU_address_array_peak','CPU address scratch peak','#ab8c63','...'),('registered_two_times_cumulative_descriptors','2× cumulative descriptor reservation','#b7b7b7','xxx')]
figvalues={}
for key,label,color,hatch in keys:
 values=np.array([sum(t['MiB'] for t in terms if t['arm']==a and t['term']==key) for a in arms]);figvalues[key]=values.tolist()
 ax.bar(positions,values,.5,bottom=bottom,color=[colors[a] for a in arms] if color is None else color,edgecolor='white' if not hatch else '#444444',linewidth=.5,hatch=hatch,label=label,zorder=3);bottom+=values
ax.axhline(512,color='#555555',linewidth=1.1,linestyle='--',zorder=4);ax.text(1.35,512,'512 MiB guard',ha='right',va='bottom',fontsize=8)
for i,v in enumerate(bottom):ax.text(i,v+7,f'{v:.2f} MiB',ha='center',va='bottom',fontsize=9,fontweight='bold')
ax.set_xticks(positions,[labels[a] for a in arms]);ax.set_xlim(-.55,1.5);ax.set_ylim(0,max(512,max(bottom))*1.19)
ax.set_ylabel('Registered scoped ownership (MiB)');ax.set_title('Measured scoped memory at 16,384 tokens',fontweight='bold',pad=10);style.despine(ax)
# Put small metadata terms in the legend rather than implying visibly large segments.
handles=[Patch(facecolor='#946a94' if color is None else color,edgecolor='#444444' if hatch else 'white',hatch=hatch,label=label) for key,label,color,hatch in keys]
ax.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False,fontsize=8)
fig.text(.5,.012,'Worst measured scoped trial per arm; every listed term is included. Both ≤512 MiB: '+('PASS' if all(v<=512 for v in bottom) else 'FAIL')+'.\nGPU allocated peak, not reserved memory. Descriptor term is a conservative reservation, not measured simultaneous usage.\nPersistent CPU root is charged idle/active; hidden host/driver/RSS excluded. These are scoped budgets, not exact whole-memory equality.',ha='center',va='bottom',fontsize=8,color='#555555',linespacing=1.3)
fig.tight_layout(rect=(0,.25,1,1))
for ext in ('pdf','png'):
 target=OUT/f'fig11_footprint.{ext}';fig.savefig(target,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {});outputs.append(target)
plt.close(fig);artists['fig11_footprint']=figvalues
captions['fig11_footprint']='Native and Snap independently charged under registered scoped512MiB formula; GPUallocatedpeak excluding model, actual264MiB pinnedCPUroot, CPUarrays/streak/address scratch, conservative2xcumulative descriptor reservation. No whole-machine/RSS equality.'
manifest={'verified_evidence':str(proofpath.relative_to(ROOT)),'verified_evidence_sha256':sha(proofpath),'raw_sha256':sha(rawpath),'generator_sha256':sha(__file__),'style_source':'experiments/plot_figures_v2.py','style_source_sha256':sha(ROOT/'experiments/plot_figures_v2.py'),'palette':colors,'data_csv_sha256':{f:sha(OUT/f) for f in ('timing_values.csv','memory_values.csv')},'outputs_sha256':{f.name:sha(f) for f in outputs},'artists':artists,'captions':captions,'registered_advancement_passed':p['advancement_passed'],'uncertainty_intervals_plotted':False,'original_figures_modified':False}
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'outputs':list(manifest['outputs_sha256']),'advancement_passed':p['advancement_passed']}))
