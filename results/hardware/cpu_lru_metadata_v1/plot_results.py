"""Standalone ordinary-backend timing and explicitly scoped metadata costs."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    files=['timestamp_tradeoff.png','timestamp_tradeoff.pdf','timestamp_tradeoff.csv','figure_manifest.json'];assert all(not (BASE/f).exists() for f in files)
    cost=read(BASE/'verified_cost.json');resource=read(BASE/'verified_resource.json');assert cost['verified'] and resource['verified']
    fig,axes=plt.subplots(1,2,figsize=(9,3.9));labels=['Ordinary bulk','CPU timestamps']
    for ax,metric,title in [(axes[0],'median_decode_ms','Median decode step'),(axes[1],'p95_decode_ms','p95 decode step')]:
        values=[cost['aggregate'][arm][metric] for arm in ['control','query']]
        bars=ax.bar(labels,values,width=.58,color=['#737e88','#bc674c']);ax.bar_label(bars,fmt='%.2f',padding=3)
        ax.set_title(title);ax.set_ylabel('End-to-end step (ms)');ax.set_ylim(0,max(values)*1.25);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    decision='whole-loop primary passed' if cost['cpu_timestamp_wholeloop_primary_passed'] else 'whole-loop primary failed'
    fig.suptitle('CPU cold timestamps: '+decision,fontsize=12)
    fig.text(.5,.87,f"Median gain {100*cost['relative_median_improvement']:.2f}% / {cost['median_decode_saved_ms']:.2f} ms; ≥5% and ≥4 ms required",ha='center',fontsize=9)
    fig.text(.025,.025,f"Quiet 3ABBA, 16K/50 steps. Extra metadata H2D 5.69 MB. Scoped active bound {resource['candidate_active_plus_cpu_metadata_tensor_bound_mib']:.2f} MiB. Research only.",fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.86]);fig.savefig(BASE/files[0],dpi=200);fig.savefig(BASE/files[1]);plt.close(fig)
    with (BASE/files[2]).open('w',newline='') as h:
        w=csv.writer(h);w.writerow(['scope','metric','ordinary_bulk','cpu_timestamps'])
        for metric in ['median_decode_ms','p95_decode_ms','median_ttft_ms']:
            w.writerow(['quiet_3abba',metric,cost['aggregate']['control'][metric],cost['aggregate']['query'][metric]])
        w.writerow(['fresh_resource','active_plus_new_CPU_tensor_bound_MiB',resource['ordinary_control_active_envelope_mib'],resource['candidate_active_plus_cpu_metadata_tensor_bound_mib']])
        w.writerow(['fresh_resource','extra_timestamp_h2d_bytes',0,resource['timestamp_h2d_bytes']])
    (BASE/files[3]).write_text(json.dumps(dict(component_proof_sha256=sha(BASE/'verified_cost.json'),resource_proof_sha256=sha(BASE/'verified_resource.json'),plot_source_sha256=sha(Path(__file__)),artifacts_sha256={f:sha(BASE/f) for f in files[:3]},limits='Separate freshresource and quiettiming. NewCPU tensorbytes included but CPUmallocrounding/otherhostretention unmeasured. No full512MiB/quality/physicalSTT/mainfaithfulbaselinewin claim.'),indent=2)+'\n')
if __name__=='__main__':main()
