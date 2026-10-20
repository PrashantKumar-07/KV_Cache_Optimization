"""Export verified component speed and unmatched faithful-baseline geometry."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 target=BASE/'figures';target.mkdir(exist_ok=False)
 speed=json.loads((BASE/'verified_cost.json').read_text());geo=json.loads((BASE/'verified_geometry.json').read_text())
 assert speed['verified'] and speed['cpu_phase_controller_wholeloop_primary_passed'] and geo['verified'] and not geo['equal_resource_P0_passed']
 labels=['Ordinary bulk','Native CPU phase'];colors=['#7b8794','#2875b9']
 fig,ax=plt.subplots(figsize=(6.2,4.2));values=[speed['aggregate'][k]['median_decode_ms'] for k in ['control','query']]
 bars=ax.bar(labels,values,color=colors,width=.55)
 for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.5,f'{value:.2f}',ha='center')
 ax.set_ylabel('Median whole-loop decode (ms/token)');ax.set_ylim(0,70);ax.set_title(f"{100*speed['relative_median_improvement']:.2f}% faster; {speed['median_decode_saved_ms']:.2f} ms saved")
 ax.spines[['top','right']].set_visible(False)
 fig.text(.5,.015,'16K/50 teacher stream; quiet3ABBA,300 forwards/arm. Ordinary control, not faithful SnapKV.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.05,1,1))
 for suffix in ['png','pdf']:fig.savefig(target/('component_speed.'+suffix),dpi=180)
 plt.close(fig)
 with (target/'component_speed.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['arm','median_ms','p95_ms','ttft_ms'])
  for arm in ['control','query']:
   r=speed['aggregate'][arm];w.writerow([arm,r['median_decode_ms'],r['p95_decode_ms'],r['median_ttft_ms']])
 fig,axes=plt.subplots(1,2,figsize=(9.6,3.8))
 axes[0].plot(range(51),[272.25]*51,color=colors[1],label='Native CPU phase (CPU+GPU)')
 axes[0].plot(range(51),[487+.5*i for i in range(51)],color='#c17827',label='Faithful SnapKV (GPU)')
 axes[0].set(xlabel='Decode boundary',ylabel='Distinct KV backing storage (MiB)',title='Unequal resident KV allocations')
 axes[0].legend(fontsize=8)
 axes[1].plot(range(1,51),[1024]*50,color=colors[1],label='Native CPU phase')
 axes[1].plot(range(1,51),range(975,1025),color='#c17827',label='Faithful SnapKV')
 axes[1].set(xlabel='Decode step',ylabel='Attended rows per layer',title='Unequal width on49/50 steps');axes[1].legend(fontsize=8)
 for ax in axes:ax.spines[['top','right']].set_visible(False)
 fig.text(.5,.01,'KV backing capacity is not complete peak memory. Baseline geometry is archived faithful data; no matched-resource or quality claim.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.05,1,1))
 for suffix in ['png','pdf']:fig.savefig(target/('unmatched_geometry.'+suffix),dpi=180)
 plt.close(fig)
 with (target/'geometry.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['boundary','candidate_KV_MiB','snap_KV_MiB','candidate_forward_rows','snap_forward_rows'])
  for i in range(51):w.writerow([i,272.25,487+.5*i,1024 if i else '',974+i if i else ''])
 manifest=dict(component_proof_sha256=sha(BASE/'verified_cost.json'),geometry_proof_sha256=sha(BASE/'verified_geometry.json'),plot_source_sha256=sha(Path(__file__)),artifacts_sha256={p.name:sha(p) for p in target.iterdir()},limits='Component speed versus ordinary control only. Geometry compares current measured candidate roots with archived faithful SnapKV at unequal allocations. No joint baseline/quality/physical STT or completion claim.')
 (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
