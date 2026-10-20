"""Stage source-backed plots without changing legacy figures or frozen experiment code.

Quality uses rescored predictions and paired target-document bootstrap. Historical
panels retain their raw source epoch and cannot establish current improvements.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from provenance import read_jsonl
from audit_unused_documents import document_group_key
from longbench_metrics import DATASET_TO_METRIC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plot_figures_v2 as style

OUT=BASE/'validated'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
INPUTS={}
def load(path):
 p=ROOT/path if not Path(path).is_absolute() else Path(path)
 INPUTS[str(p.relative_to(ROOT))]=sha(p)
 return json.loads(p.read_text())

def quality(name):
 directory=Path('results/audit_20261013')/name
 proof=load(directory/'verified_quality.json')
 assert proof['verified']
 if name=='final_prior_development':
  paths=[directory/'qa.json',directory/'retrieval_summary.json'];methods=['h2o','snapkv','tieredkv']
 else:
  paths=[directory/'snap.json',directory/'tiered.json'];methods=['snapkv_upstream_selector','tieredkv']
 raw=[load(p) for p in paths]
 for p in paths:
  expected=proof['evidence_sha256'].get(str(p),proof['evidence_sha256'].get(p.name))
  assert sha(ROOT/p)==expected,p
 results={};configs={}
 for doc in raw:
  for task,arms in doc['results'].items():
   results.setdefault(task,{}).update(arms)
   for method in arms:configs[task,method]=doc['config']
 rng=np.random.default_rng(0);summary={'methods':methods,'tasks':{},'rescored_predictions':0,'bootstrap_draws':10000,'bootstrap_seed':0}
 overall={m:[] for m in methods};overall_diff=[]
 for task,arms in results.items():
  cfg=configs[task,methods[0]]
  directory_data=ROOT/cfg['data_dir']
  path=next(p for p in [directory_data/f'{task}_e.jsonl',directory_data/f'{task}.jsonl'] if p.exists())
  assert sha(path)==cfg['dataset_sha256'][task]
  INPUTS[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)]=sha(path)
  source=read_jsonl(path);lookup={hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest():x for x in source}
  ids=[x['sample_id'] for x in arms[methods[0]]['samples']];assert len(ids)==5==len(set(ids))
  groups={};scores={m:[] for m in methods}
  for i,sid in enumerate(ids):
   item=lookup[sid];groups.setdefault(document_group_key(item,task),[]).append(i)
   ref=arms[methods[0]]['samples'][i]
   for method in methods:
    row=arms[method]['samples'][i]
    for key in ['sample_id','prompt_sha256','input_ids_sha256']:
     assert row[key]==ref[key],(task,method,key)
    pred=row['prediction']
    if task in ('triviaqa','trec','samsum','lsht'):pred=pred.lstrip('\n').split('\n')[0]
    score=100*max(DATASET_TO_METRIC[task](pred,a,all_classes=item.get('all_classes',[])) for a in item['answers'])
    assert abs(score-row['score'])<1e-9
    scores[method].append(score);summary['rescored_predictions']+=1
  indices=list(groups.values());sizes=np.array([len(g) for g in indices]);draws=rng.integers(0,len(indices),(10000,len(indices)))
  taskrow={'sample_ids':ids,'target_document_groups':list(groups),'group_row_indices':indices,'methods':{}}
  boot={}
  for method in methods:
   values=np.asarray(scores[method]);sums=np.array([values[g].sum() for g in indices]);boot[method]=sums[draws].sum(1)/sizes[draws].sum(1)
   overall[method].append(boot[method]);taskrow['methods'][method]={'scores':scores[method],'mean':float(values.mean()),'cluster_bootstrap_ci95':np.quantile(boot[method],[.025,.975]).tolist()}
  baseline='snapkv' if 'snapkv' in methods else 'snapkv_upstream_selector'
  delta=boot['tieredkv']-boot[baseline];overall_diff.append(delta)
  taskrow['tiered_minus_snap']={'mean':float(np.mean(scores['tieredkv'])-np.mean(scores[baseline])),'cluster_bootstrap_ci95':np.quantile(delta,[.025,.975]).tolist()}
  summary['tasks'][task]=taskrow
 summary['task_balanced_means']={m:float(np.mean([t['methods'][m]['mean'] for t in summary['tasks'].values()])) for m in methods}
 summary['task_balanced_tiered_minus_snap']={'mean':float(np.mean([t['tiered_minus_snap']['mean'] for t in summary['tasks'].values()])), 'cluster_bootstrap_ci95':np.quantile(np.mean(overall_diff,axis=0),[.025,.975]).tolist()}
 if name=='final_prior_development':
  assert all(abs(summary['task_balanced_means'][m]-proof['task_balanced_means'][m])<1e-9 for m in methods)
 else:
  assert np.allclose(summary['task_balanced_tiered_minus_snap']['cluster_bootstrap_ci95'],proof['within_task_target_document_cluster_bootstrap_95'],rtol=0,atol=1e-12)
 summary['source_epochs']=sorted({d['config']['runtime']['source_sha256'] for d in raw})
 summary['scope']='Consumed development, six fixed tasks, one model/seed, 5rows/task; target-document bootstrap conditional on sampled groups. No independent quality or superiority claim.'
 return summary

SAVED=[]
ORIGINAL_SAVE=Figure.savefig
CURRENT={}
def save(self,fname,*args,**kwargs):
 name=Path(fname).name
 if name.endswith('.png'):
  SAVED.append({'file':name,'question':CURRENT.get('question'),'scope':CURRENT.get('scope'),'input_paths':list(CURRENT.get('inputs',[])),
   'axes':[{'title':ax.get_title(),'ylabel':ax.get_ylabel(),'ylim':list(ax.get_ylim()),'xticklabels':[x.get_text() for x in ax.get_xticklabels()],
    'bars':[{'x':p.get_x(),'y':p.get_y(),'height':p.get_height()} for p in ax.patches if hasattr(p,'get_height')],
    'lines':[{'x':np.asarray(l.get_xdata()).tolist(),'y':np.asarray(l.get_ydata()).tolist()} for l in ax.lines],
    'images':[np.asarray(im.get_array()).tolist() for im in ax.images], 'texts':[t.get_text() for t in ax.texts]} for ax in self.axes],
   'figure_texts':[t.get_text() for t in self.texts]})
 return ORIGINAL_SAVE(self,OUT/name,*args,**kwargs)

def finish(fig,stem):
 for ext in ['png','pdf']:fig.savefig(OUT/f'{stem}.{ext}')
 plt.close(fig)

def bars(summary,stem,title,note,labels):
 methods=summary['methods'];tasks=list(summary['tasks']);width=.68/len(methods);x=np.arange(len(tasks))
 fig,ax=plt.subplots(figsize=(9.6,4.2))
 for j,m in enumerate(methods):
  values=np.array([summary['tasks'][t]['methods'][m]['mean'] for t in tasks]);intervals=np.array([summary['tasks'][t]['methods'][m]['cluster_bootstrap_ci95'] for t in tasks])
  color=style.FIG1_PALETTE.get(m,'#AEB6BF');offset=(j-(len(methods)-1)/2)*width
  ax.bar(x+offset,values,width*.92,label=labels[m],color=color,edgecolor='white',linewidth=.7,hatch='///' if 'snap' in m else '',yerr=np.maximum(0,np.stack([values-intervals[:,0],intervals[:,1]-values])),error_kw={'capsize':2,'elinewidth':.7,'ecolor':'#333333'},zorder=3)
  for i,v in enumerate(values):ax.text(x[i]+offset,intervals[i,1]+1.5,f'{v:.1f}',ha='center',va='bottom',rotation=90,fontsize=7,fontweight='bold')
 ax.set_xticks(x,[style.TASK_LABELS[t] for t in tasks],fontsize=9);ax.set_ylim(0,118)
 ax.set_ylabel('F1 / ROUGE-L (points)',fontsize=10);ax.set_title(title,fontsize=10.5,fontweight='bold',pad=6)
 ax.legend(ncol=len(methods),loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False,fontsize=8)
 style.despine(ax);fig.text(.5,-.01,note,ha='center',fontsize=7,color='#555555');fig.tight_layout(rect=(0,.09,1,1));finish(fig,stem)

def main():
 global CURRENT
 OUT.mkdir(exist_ok=False)
 Figure.savefig=save
 adapted=quality('final_prior_development');faithful=quality('faithful_snap_development')
 (OUT/'quality_data.json').write_text(json.dumps({'adapted':adapted,'faithful':faithful},indent=2))
 CURRENT={'question':'Adapted-baseline development quality','scope':adapted['scope'],'inputs':['results/audit_20261013/final_prior_development/qa.json','results/audit_20261013/final_prior_development/retrieval_summary.json']}
 d=adapted['task_balanced_tiered_minus_snap']
 bars(adapted,'fig1_accuracy_equivram','Development quality: adapted baselines, equal hot KV capacity',
  f"5 consumed rows/task; extra Tiered cold capacity/attention. Error bars: 95% target-document bootstrap.\nTiered−adapted Snap: {d['mean']:+.2f} points, cluster 95% [{d['cluster_bootstrap_ci95'][0]:.2f}, {d['cluster_bootstrap_ci95'][1]:.2f}]. No superiority established.",
  {'h2o':'Adapted H2O','snapkv':'Adapted SnapKV','tieredkv':'TieredKV proxy'})
 CURRENT={'question':'Faithful-selector development quality','scope':faithful['scope'],'inputs':['results/audit_20261013/faithful_snap_development/snap.json','results/audit_20261013/faithful_snap_development/tiered.json']}
 d=faithful['task_balanced_tiered_minus_snap'];note=f"5 consumed rows/task; initial long-prompt KV payload boundary only (452 MiB), excludes metadata/temporaries.\nError bars: 95% target-document bootstrap; Tiered−Snap {d['mean']:+.2f}, 95% [{d['cluster_bootstrap_ci95'][0]:.2f}, {d['cluster_bootstrap_ci95'][1]:.2f}]. Allocation/width/decode trajectories differ."
 bars(faithful,'fig7_accuracy_fair','Development quality: faithful SnapKV selector versus actual-query TieredKV',note,{'snapkv_upstream_selector':'Faithful SnapKV selector','tieredkv':'TieredKV actual-query'})
 tasks=list(faithful['tasks']);methods=faithful['methods'];mat=np.array([[faithful['tasks'][t]['methods'][m]['mean'] for m in methods] for t in tasks])
 fig,ax=plt.subplots(figsize=(8.5,4.6));im=ax.imshow(mat,cmap='YlGn',aspect='auto',vmin=0,vmax=100)
 ax.set_xticks(range(2),['Faithful SnapKV selector','TieredKV actual-query'],fontsize=9);ax.set_yticks(range(6),[style.TASK_LABELS[t] for t in tasks],fontsize=9)
 for i in range(6):
  for j in range(2):
   best=mat[i,j]==max(mat[i]);ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=9,fontweight='bold' if best else 'normal',bbox=dict(boxstyle='round,pad=.2',fc='white',ec='none',alpha=.85) if best else None)
 ax.set_title('Consumed development scores (boxed = tied or best point estimate)',fontsize=11,fontweight='bold');fig.colorbar(im,ax=ax,shrink=.9).set_label('F1 / ROUGE-L (points)')
 fig.text(.5,.005,'Same 30 pairs as Fig. 7; initial KV payload boundary only. Boxes do not indicate statistical significance.',ha='center',fontsize=7);fig.tight_layout(rect=(0,.05,1,1));finish(fig,'fig14_heatmap')
 # Historical panels use their exact raw JSON values and retain visible limits.
 import plot_publication_figures as pub
 fairpath='results/final_equal_attended_frac075.json';evpath='results/final_equal_vram_frac075.json'
 fair,ev=load(fairpath),load(evpath)
 old_save=pub.savefig
 def historic_save(fig,outdir,stem):
  if stem=='fig2_latency_comparison':
   fig.axes[0].set_title('HISTORICAL: summed modeled decode traffic cost (adapted baselines)',fontsize=10.5,fontweight='bold')
   fig.axes[0].set_ylabel('Summed modeled decode traffic time (ms)');fig.texts[0].set_text('Unequal free-generation output lengths; not per-token latency or measured speed. Old protocol; no source/sample identity hashes.')
  elif stem=='fig3_write_savings':
   fig.axes[0].set_ylabel('Logical per-layer token demotion events')
   fig.axes[0].set_title('Historical: paid and avoided demotions (fraction 0.75)',fontsize=10)
   fig.axes[1].set_title('Historical: separate fraction 1.0 budget sweep',fontsize=10)
   fig.axes[1].set_ylabel('Unweighted mean task savings (%)')
   fig.texts[0].set_text('HISTORICAL: different fractions and task scopes across panels')
   for text in fig.axes[1].texts:
    if text.get_text().startswith('Higher STT'):text.set_text('Five tasks; task ratios averaged. Decode demotions only; not physical pages.')
  elif stem=='fig6_migration_profile':
   fig.axes[0].set_ylabel('Logical per-layer event counters (sum)');fig.axes[0].set_title('HISTORICAL: logical policy actions, including non-transfer deferrals',fontsize=10.5,fontweight='bold')
   fig.text(.5,.005,'Unequal output lengths; deferrals perform no migration. Old protocol; no source/sample identity hashes.',ha='center',fontsize=7)
  for ext in ['png','pdf']:fig.savefig(OUT/f'{stem}.{ext}')
  plt.close(fig)
 pub.savefig=historic_save
 CURRENT={'question':'Historical analytical workload costs','scope':'Legacy fraction0.75; output lengths differ; not measured or per-token latency','inputs':[fairpath]};pub.fig2_latency(fair,str(OUT))
 sweeps=sorted(Path('results/clean_sweep_stt').glob('stt*.json'))
 for p in sweeps:load(p)
 CURRENT={'question':'Historical logical avoided demotions','scope':'Left fraction.75/six tasks; right fraction1/five tasks. Per-layer token events, not pages/endurance.','inputs':[fairpath,*map(str,sweeps)]};pub.fig3_write_savings(fair,ev,sweeps,str(OUT))
 CURRENT={'question':'Historical policy event composition','scope':'Logical actions including non-transfer deferrals; not bytes','inputs':[evpath]};pub.fig6_migration_profile(ev,str(OUT))
 # Restore v2 style exactly, then annotate its historical panels.
 import importlib
 importlib.reload(style);style.OUT=str(OUT)
 original_v2_save=style.savefig
 def v2_save(fig,stem):
  ax=fig.axes[0]
  if stem=='fig4_decode_scaling':
   ax.set_title('HISTORICAL: full-loop decode timing, simulated tiers on GPU',fontsize=11)
   fig.texts[0].set_text('One GovReport prefix; 3 repeats × 50 greedy decode steps, EOS ignored. Adapted baselines; no controlled speedup claim.')
  elif stem=='fig11_footprint':
   ax.set_title('HISTORICAL configured capacities: analytical BF16 GQA payload only',fontsize=11)
   fig.text(.5,.005,'32 layers × 8 KV heads × 128 dimensions × K/V × 2 bytes. Excludes duplicates, metadata and allocator overhead.',ha='center',fontsize=7)
  elif stem=='fig12_energy':
   ax.set_title('HISTORICAL: analytical decode KV-traffic energy at assumed prices',fontsize=11)
   fig.texts[0].set_text('Unequal output lengths; assumed STT 2/8 and VRAM 12 pJ/bit. Residual is other billed traffic, not isolated attention.')
   handles,labels=ax.get_legend_handles_labels();labels[0]='Other billed\ntraffic (residual)';ax.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.15))
  finish(fig,stem)
 style.savefig=v2_save
 lp='results/remediation_v3/latency.json';lat=load(lp)
 for method,rows in lat['results'].items():
  for ctx,row in rows.items():assert round(float(np.median([s for trial in row['trials'] for s in trial['step_ms']])),3)==row['itl_ms']
 CURRENT={'question':'Historical context scaling','scope':'Observed medians only; old adapted/equalhot source epoch','inputs':[lp]};style.fig4_decode_scaling(lat)
 ev3path='results/remediation_v3/equal_vram_merged.json';ev3=load(ev3path)
 CURRENT={'question':'Configured theoretical KV capacity','scope':'Not measured allocation, occupancy or current native/faithful resources','inputs':[ev3path]};style.fig11_footprint(ev3)
 CURRENT={'question':'Historical assumed traffic energy','scope':'Modeled totals; unequal work; residual not pure attention; not system energy','inputs':[ev3path,'src/cost_model.py']};INPUTS['src/cost_model.py']=sha(ROOT/'src/cost_model.py');style.fig12_energy(ev3)
 # Same old fraction-study style and numeric series; remove unsupported NI claims.
 import plot_frac_ablation as frac
 paths=sorted(Path('results').glob('e2_frac*_short.json'))
 for p in paths:load(p)
 rows=sorted(frac.load_frac(str(p)) for p in paths)
 CURRENT={'question':'Historical headroom sweep','scope':'Five consumed tasks,25rows each; unweighted task ratios; no uncertainty/NI proof or source hashes','inputs':list(map(str,paths))}
 fig,ax=plt.subplots(figsize=(7.5,4.2));x=[r[0] for r in rows];ax.plot(x,[r[1] for r in rows],'o-',color='#1d3557');ax.invert_xaxis();ax.set_xlabel('STT-RAM bifurcation fraction (headroom ←)');ax.set_ylabel('Mean task F1 (points)',color='#1d3557')
 ax2=ax.twinx();ax2.plot(x,[r[2] for r in rows],'s--',color='#27ae60');ax2.set_ylabel('Mean task avoided-demotion ratio (%)',color='#27ae60');ax2.set_ylim(0,100)
 for f,s,v in rows:ax.annotate(f'{s:.1f}',(f,s),textcoords='offset points',xytext=(0,8),ha='center',fontsize=8)
 ax.set_title('HISTORICAL: headroom sweep point estimates\nAccuracy non-inferiority was not established',fontsize=10.5,fontweight='bold');fig.text(.5,.005,'Five tasks ×25 rows; old prompt/scoring protocol. Task percentages averaged; prompt-placement writes excluded.',ha='center',fontsize=7);fig.tight_layout(rect=(0,.05,1,1));finish(fig,'fig8_frac_ablation')
 import plot_frac_compare as comp
 p1='results/e2_frac1_0_short.json';p75='results/e2_frac0_75_short.json';pbase='results/final_equal_vram.json';pgov='results/frac075_gov.json'
 for p in [p1,p75,pbase,pgov]:load(p)
 CURRENT={'question':'Historical paired fraction point estimates','scope':'Not current six-task quality; no NI proof/sample identity; unequal output work','inputs':[p1,p75,pbase,pgov]}
 # Capture and relabel immediately before save; underlying series/layout unchanged.
 outer=Figure.savefig
 def fracsave(self,fname,*args,**kwargs):
  if 'fig15_frac_compare' in str(fname):
   self.axes[0].set_title('(a) Historical score point estimates',fontweight='bold');self.axes[1].set_title('(b) Historical avoided-demotion ratios',fontweight='bold');self.axes[1].set_ylabel('Logical decode demotions avoided (%)')
   self._suptitle.set_text('HISTORICAL: fraction 1.0 versus 0.75; no accuracy non-inferiority claim')
   if not any(t.get_text().startswith('Five tasks') for t in self.texts):self.text(.5,-.025,'Five tasks ×25 rows; GovReport ×10. Old protocol; excludes prompt placement; not physical page writes.',ha='center',fontsize=7)
  return outer(self,fname,*args,**kwargs)
 Figure.savefig=fracsave;comp.main();Figure.savefig=outer
 for item in SAVED:
  item['output_sha256']={ext:sha(OUT/Path(item['file']).with_suffix('.'+ext).name) for ext in ['png','pdf']}
 report={'staged_only':True,'script_sha256':sha(__file__),'quality':{'adapted':adapted,'faithful':faithful},'input_sha256':INPUTS,'figures':SAVED,
  'limits':'Three panels updated to verified development epochs; eight explicitly historical or analytical. No new model evaluation, independent-quality claim, current fair-timing result, or promotion.'}
 (OUT/'provenance.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
 assert len(SAVED)==11 and len(set(x['file'] for x in SAVED))==11
 print('Staged 11 pairs; rescored150 predictions; paired group uncertainty recomputed.')

if __name__=='__main__':main()
