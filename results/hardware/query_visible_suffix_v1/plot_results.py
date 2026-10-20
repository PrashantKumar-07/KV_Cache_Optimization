"""Standalone figure with separate component and ordinary-LRU comparisons."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    names=['suffix_tradeoff.png','suffix_tradeoff.pdf','suffix_tradeoff.csv','figure_manifest.json']
    assert all(not (BASE/name).exists() for name in names)
    component=json.loads((BASE/'verified_cost.json').read_text())
    guard=json.loads((BASE/'verified_lru.json').read_text())
    assert component['verified'] and guard['verified']
    gain=100*component['relative_median_improvement']
    change=100*guard['relative_median_change']
    direction='slower' if change>=0 else 'faster'
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    panels=[(axes[0],component,['Query concat','Query suffix writes'],'Composition component',f'{gain:.2f}% faster than query concat'),
            (axes[1],guard,['Ordinary LRU','Query suffix writes'],'Complete candidate guard',f'{abs(change):.2f}% {direction} than ordinary LRU')]
    for ax,report,labels,title,note in panels:
        values=[report['aggregate'][arm]['median_decode_ms'] for arm in ('control','query')]
        bars=ax.bar(labels,values,color=['#737e88','#bc674c'],width=.58)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=10)
        ax.set_title(title);ax.set_ylim(0,130)
        ax.text(.5,.94,note,transform=ax.transAxes,ha='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    axes[0].set_ylabel('Median end-to-end decode step (ms)')
    outcome='latency guard passed' if guard['ordinary_lru_regression_guards_passed'] else 'latency guard failed'
    fig.suptitle('Cold-suffix composition: component gain, '+outcome,fontsize=12)
    fig.text(.025,.025,'Separate quiet 3ABBA, 16K/50-step comparisons. KV H2D 5.826 GB; index copies 75.33 MB. Research only.',fontsize=8)
    fig.tight_layout(rect=[0,.065,1,.91])
    fig.savefig(BASE/names[0],dpi=200);fig.savefig(BASE/names[1]);plt.close(fig)
    with (BASE/names[2]).open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['comparison','metric','control','query_suffix_writes'])
        for name,report in [('query_concat_component',component),('ordinary_lru_guard',guard)]:
            for metric in ('median_decode_ms','p95_decode_ms','median_ttft_ms'):
                writer.writerow([name,metric,report['aggregate']['control'][metric],report['aggregate']['query'][metric]])
    (BASE/names[3]).write_text(json.dumps(dict(component_proof_sha256=sha(BASE/'verified_cost.json'),
        lru_proof_sha256=sha(BASE/'verified_lru.json'),plot_source_sha256=sha(Path(__file__)),
        artifacts_sha256={name:sha(BASE/name) for name in names[:3]},
        limits='Separate paired comparisons, one model/seed. No totalH2D saving, independent quality, full512MiB, matched faithful-baseline or physicalSTT claim.'),indent=2)+'\n')
if __name__=='__main__':main()
