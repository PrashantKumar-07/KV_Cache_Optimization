"""Export measured component outcomes with comparison limits."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=json.loads((BASE/'verified_cost.json').read_text());resource=json.loads((BASE/'verified_resources.json').read_text());snap=json.loads((BASE.parents[0]/'prompt_boundary_comparison_v1/snap.json').read_text())
    out=BASE/'figures';out.mkdir(exist_ok=True);assert not (out/'manifest.json').exists()
    rows=[dict(arm=arm,median_ms=result['summary'][arm]['median_ms'],p95_ms=result['summary'][arm]['p95_ms'],prefill_gb=snap['gpu_phase_peaks']['prefill_peak']/1e9 if arm=='snap' else resource['prefill_peak_gb']) for arm in ['snap','cpu']]
    with (out/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(8,4));labels=['Faithful SnapKV','Early shared CPU'];colors=['#437bb1','#cc7b37']
    for ax,key,title in [(axes[0],'median_ms','Median decode (ms/token)'),(axes[1],'prefill_gb','Fresh prefill peak (GB)')]:
        values=[r[key] for r in rows];ax.bar(labels,values,color=colors);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
        for i,value in enumerate(values):ax.text(i,value,f'{value:.2f}',ha='center',va='bottom')
        ax.set_ylim(0,max(values)*1.17)
    fig.text(.5,.015,'One model / seed / 16K prompt; 300 decode forwards per arm. Initial KV equal; full trajectory unequal.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.06,1,1])
    for extension in ['png','pdf']:fig.savefig(out/('comparison.'+extension),dpi=180)
    plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(dict(verified=True,source_proofs={p:sha(BASE/p) for p in ['verified_cost.json','verified_resources.json']},files={p.name:sha(p) for p in out.iterdir()},plotter_sha256=sha(Path(__file__)),full_goal_complete=False),indent=2)+'\n')
if __name__=='__main__':main()
