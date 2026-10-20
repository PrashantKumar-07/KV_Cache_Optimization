"""Standalone measured ownership memory and complete-loop timing artifacts."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    files=['ownership_tradeoff.png','ownership_tradeoff.pdf','ownership_tradeoff.csv','figure_manifest.json']
    assert all(not (BASE/f).exists() for f in files)
    cost=read(BASE/'verified_cost.json');resource=read(BASE/'verified_resource.json')
    assert cost['verified'] and resource['verified']
    fig,axes=plt.subplots(1,3,figsize=(11.4,4.2));labels=['Ordinary bulk','Pooled controller']
    values=[resource['ordinary_active_mib'],resource['candidate_active_mib']]
    for ax,vals,title,ylabel in [(axes[0],values,'Scoped active memory','MiB'),(axes[1],[cost['aggregate'][a]['median_decode_ms'] for a in ['control','query']],'Median decode step','End-to-end step (ms)'),(axes[2],[cost['aggregate'][a]['p95_decode_ms'] for a in ['control','query']],'p95 decode step','End-to-end step (ms)')]:
        bars=ax.bar(labels,vals,width=.58,color=['#737e88','#397a87']);ax.bar_label(bars,fmt='%.2f',padding=3)
        ax.set_title(title);ax.set_ylabel(ylabel);ax.set_ylim(0,max(vals)*1.25);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True);ax.tick_params(axis='x',labelsize=8)
    axes[0].axhline(512,color='#a74632',ls='--',lw=1,label='512 MiB gate');axes[0].legend(fontsize=8)
    decision='passed' if cost['pooled_controller_wholeloop_primary_passed'] else 'failed'
    fig.suptitle('Pooled controller: memory passed; timing '+decision,fontsize=12)
    fig.text(.5,.875,f"Memory reduction {100*resource['relative_active_reduction']:.2f}%; median speed gain {100*cost['relative_median_improvement']:.2f}% / {cost['median_decode_saved_ms']:.2f} ms saved",ha='center',fontsize=9)
    fig.text(.02,.027,'Separate fresh memory audit and quiet 3ABBA, 16K/50 steps. Owned pinned block included; constant model excluded. Research only.',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.86]);fig.savefig(BASE/files[0],dpi=200);fig.savefig(BASE/files[1]);plt.close(fig)
    with (BASE/files[2]).open('w',newline='') as h:
        w=csv.writer(h);w.writerow(['scope','metric','ordinary_bulk','pooled_controller'])
        w.writerow(['fresh_resource','active_MiB',*values])
        for metric in ['median_decode_ms','p95_decode_ms','median_ttft_ms']:w.writerow(['quiet_3abba',metric,cost['aggregate']['control'][metric],cost['aggregate']['query'][metric]])
    (BASE/files[3]).write_text(json.dumps(dict(component_proof_sha256=sha(BASE/'verified_cost.json'),resource_proof_sha256=sha(BASE/'verified_resource.json'),plot_source_sha256=sha(Path(__file__)),artifacts_sha256={f:sha(BASE/f) for f in files[:3]},limits='Separate resource and timing runs, one Mistral/teacher/seed. Scoped active allocation only; not full-process memory, independent quality, faithful baseline or physical STT proof.'),indent=2)+'\n')
if __name__=='__main__':main()
