"""Two separate verified comparisons: component gain and ordinaryLRU failure."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    names=['read_index_tradeoff.png','read_index_tradeoff.pdf','read_index_tradeoff.csv','figure_manifest.json']
    assert all(not (BASE/n).exists() for n in names)
    component=json.loads((BASE/'verified_cost.json').read_text());guard=json.loads((BASE/'verified_lru.json').read_text())
    assert component['verified'] and guard['verified']
    fig,axes=plt.subplots(1,2,figsize=(8.7,3.7),sharey=True)
    for ax,report,labels,title,note in [(axes[0],component,['Original query','GPU read indices'],'Read-index component','24.61% faster than original query'),
            (axes[1],guard,['Ordinary LRU','GPU read indices'],'Complete candidate guard','80.99% slower than ordinary LRU')]:
        values=[report['aggregate'][arm]['median_decode_ms'] for arm in ('control','query')]
        bars=ax.bar(labels,values,color=['#737e88','#bc674c'],width=.58)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=10)
        ax.set_title(title);ax.set_ylim(0,165);ax.text(.5,.94,note,transform=ax.transAxes,ha='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    axes[0].set_ylabel('Median end-to-end decode step (ms)')
    fig.suptitle('Native GPU read indices: component gain, overall latency failure',fontsize=12)
    fig.text(.025,.025,'Separate quiet3ABBA16K/50-step comparisons. KV reads unchanged:5.826GB; candidate index copies75.33MB. Research-only.',fontsize=8)
    fig.tight_layout(rect=[0,.065,1,.91]);fig.savefig(BASE/names[0],dpi=200);fig.savefig(BASE/names[1]);plt.close(fig)
    with (BASE/names[2]).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['comparison','metric','control','gpu_read_indices'])
        for name,report in [('original_query_component',component),('ordinary_lru_guard',guard)]:
            for metric in ('median_decode_ms','p95_decode_ms','median_ttft_ms'):
                writer.writerow([name,metric,report['aggregate']['control'][metric],report['aggregate']['query'][metric]])
    (BASE/names[3]).write_text(json.dumps(dict(component_proof_sha256=sha(BASE/'verified_cost.json'),lru_proof_sha256=sha(BASE/'verified_lru.json'),
        plot_source_sha256=sha(Path(__file__)),artifacts_sha256={n:sha(BASE/n) for n in names[:3]},
        limits='Separate paired comparisons, one model/seed. No totalH2D saving, independent quality, full512MiB, matched faithful-baseline or physicalSTT claim.'),indent=2)+'\n')
if __name__=='__main__':main()
