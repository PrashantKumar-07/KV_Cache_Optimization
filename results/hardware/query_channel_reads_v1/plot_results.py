"""Standalone resource/latency figure; separate measurements and scopes."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    files=['channel_tradeoff.png','channel_tradeoff.pdf','channel_tradeoff.csv','figure_manifest.json']
    assert all(not (BASE/f).exists() for f in files)
    resource=read(BASE/'verified_resource.json');cost=read(BASE/'verified_cost.json')
    assert resource['verified'] and cost['verified']
    fig,axes=plt.subplots(1,2,figsize=(9,3.9))
    labels=['Paired K/V reads','Keys + selected values']
    payload=[resource[k]/1e9 for k in ['control_copied_kv_h2d_bytes','candidate_copied_kv_h2d_bytes']]
    latency=[cost['aggregate'][a]['median_decode_ms'] for a in ['control','query']]
    for ax,values,title,ylabel,ylim in [(axes[0],payload,'Native copied K/V payload','H2D K/V payload per 50-step trial (GB)',7),
                                       (axes[1],latency,'Quiet 3ABBA component timing','Median end-to-end decode step (ms)',max(latency)*1.25)]:
        bars=ax.bar(labels,values,color=['#737e88','#bc674c'],width=.58)
        ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_title(title);ax.set_ylabel(ylabel);ax.set_ylim(0,ylim)
        ax.tick_params(axis='x',labelsize=8);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    outcome='component primary passed' if cost['channel_component_primary_passed'] else 'component primary failed'
    fig.suptitle('Exact channel reads: less copied payload, '+outcome,fontsize=11)
    axes[0].text(.5,.94,f"{100*resource['copied_kv_payload_reduction']:.2f}% less copied K/V",transform=axes[0].transAxes,ha='center',fontsize=9)
    axes[1].text(.5,.94,f"{100*cost['relative_median_improvement']:.2f}% median gain; ≥5% and ≥4 ms required",transform=axes[1].transAxes,ha='center',fontsize=9)
    fig.text(.025,.025,'16K/50 steps, one model/seed. Active 545.17 MiB; index H2D 75.33 MB. CPU DRAM surrogate; research only.',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.90]);fig.savefig(BASE/files[0],dpi=200);fig.savefig(BASE/files[1]);plt.close(fig)
    with (BASE/files[2]).open('w',newline='') as h:
        w=csv.writer(h);w.writerow(['scope','metric','paired_reader','channel_reader'])
        w.writerow(['fresh_resource','copied_kv_h2d_bytes',resource['control_copied_kv_h2d_bytes'],resource['candidate_copied_kv_h2d_bytes']])
        for metric in ['median_decode_ms','p95_decode_ms','median_ttft_ms']:
            w.writerow(['component_3abba',metric,cost['aggregate']['control'][metric],cost['aggregate']['query'][metric]])
    (BASE/files[3]).write_text(json.dumps(dict(resource_proof_sha256=sha(BASE/'verified_resource.json'),component_proof_sha256=sha(BASE/'verified_cost.json'),plot_source_sha256=sha(Path(__file__)),artifacts_sha256={f:sha(BASE/f) for f in files[:3]},limits='Copied nativeKV payload excludes library H2D and physicalSTT/PCIe. Separate fresh-resource and quiet3ABBA measurements; no independentquality/full512MiB/adoption claim.'),indent=2)+'\n')
if __name__=='__main__':main()
