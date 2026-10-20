"""Standalone diagnostic figure from verified development and timing gates."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    names=['query_tradeoff.png','query_tradeoff.pdf','query_tradeoff.csv','figure_manifest.json']
    assert all(not (BASE/n).exists() for n in names)
    dev=json.loads((BASE/'verified_development.json').read_text())
    timing=json.loads((BASE/'verified_timing.json').read_text())
    assert dev['verified'] and timing['verified']
    tasks=list(dev['per_task']);deltas=[dev['per_task'][t]['mean_score_difference'] for t in tasks]
    labels=['Qasper','NarrativeQA','MultiFieldQA','HotpotQA','TriviaQA','GovReport']
    figure,axes=plt.subplots(1,2,figsize=(9.2,3.8),gridspec_kw={'width_ratios':[1.3,1]})
    axes[0].barh(labels,deltas,color='#3e7189');axes[0].axvline(0,color='#555555',linewidth=.8)
    axes[0].invert_yaxis();axes[0].set_xlabel('Query − LRU score points');axes[0].set_title('Reused development: 12 paired requests')
    for i,value in enumerate(deltas):axes[0].text(value+.12,i,f'{value:+.2f}',va='center',fontsize=9)
    axes[0].set_xlim(-.25,max(deltas)+1.6)
    x=np.arange(2);width=.34
    for offset,arm,color,label in [(-width/2,'control','#737e88','Ordinary LRU'),(width/2,'query','#bc674c','Current query')]:
        values=[timing['aggregate'][arm]['median_decode_ms'],timing['aggregate'][arm]['p95_decode_ms']]
        bars=axes[1].bar(x+offset,values,width,color=color,label=label)
        axes[1].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    axes[1].set_xticks(x,['Median','p95']);axes[1].set_ylabel('End-to-end decode step (ms)')
    axes[1].set_title('16K/50 steps, three ABBA blocks');axes[1].legend(loc='upper left',frameon=False,fontsize=9)
    axes[1].set_ylim(0,max(timing['aggregate']['query']['p95_decode_ms'],timing['aggregate']['control']['p95_decode_ms'])*1.55)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y' if ax is axes[1] else 'x',alpha=.12);ax.set_axisbelow(True)
    figure.suptitle('Native pool current-query selector: research tradeoff',fontsize=13)
    figure.text(.03,.025,'Development does not establish independent quality. Full scan: 5.826 GB H2D / 50 steps. Ordinary adoption remains disabled.',fontsize=8)
    figure.tight_layout(rect=[0,.075,1,.93])
    figure.savefig(BASE/names[0],dpi=200);figure.savefig(BASE/names[1]);plt.close(figure)
    with (BASE/names[2]).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['scope','item','control','query','query_minus_control'])
        for task,value in zip(tasks,deltas):writer.writerow(['development_score_points',task,'','',value])
        for metric in ('median_decode_ms','p95_decode_ms','median_ttft_ms'):
            a,b=(timing['aggregate'][arm][metric] for arm in ('control','query'));writer.writerow(['timing',metric,a,b,b-a])
    manifest=dict(development_proof_sha256=sha(BASE/'verified_development.json'),timing_proof_sha256=sha(BASE/'verified_timing.json'),
        plot_source_sha256=sha(Path(__file__)),artifacts_sha256={n:sha(BASE/n) for n in names[:3]},
        limits='Research diagnostic; two reused requests/task, one model/seed. No independent quality, full512MiB, actual matched faithful-baseline resources, physicalSTT or adoption proof.')
    (BASE/names[3]).write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
