"""Diagnostic only: verified offline recall screen, not measured H2D savings."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    names=['recall_screen.png','recall_screen.pdf','recall_screen.csv','figure_manifest.json']
    assert all(not (BASE/n).exists() for n in names)
    report=json.loads((BASE/'verified_report.json').read_text());assert report['verified']
    layer=list(range(32));recall=[100*report['per_layer'][str(i)]['mean_top32_overlap'] for i in layer]
    mass=[100*report['per_layer'][str(i)]['mean_full_oracle_mass_ceiling'] for i in layer]
    fig,ax=plt.subplots(figsize=(8,3.6))
    ax.plot(layer,recall,marker='o',markersize=3,label='Top-32 row overlap',color='#3e7189')
    ax.plot(layer,mass,marker='s',markersize=3,label='Best possible selected mass',color='#bc674c')
    ax.axhline(90,color='#3e7189',linestyle='--',linewidth=1,label='90% row-overlap screen')
    ax.axhline(95,color='#bc674c',linestyle=':',linewidth=1,label='95% mass screen')
    ax.set(xlabel='Mistral attention layer',ylabel='Mean retained (%)',ylim=(0,103),xlim=(-.5,31.5))
    ax.set_title('Fixed eight centroid pages: native query trajectory, 16K / 50 steps')
    ax.legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.21),frameon=False)
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.text(.025,.025,'Offline full-key rebuild; candidates cover 13.5% of live rows. No measured H2D savings, maintained-index cost or quality claim.',fontsize=8)
    fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(BASE/names[0],dpi=200);fig.savefig(BASE/names[1]);plt.close(fig)
    with (BASE/names[2]).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['layer','mean_top32_overlap_pct','mean_full_oracle_mass_ceiling_pct'])
        writer.writerows(zip(layer,recall,mass))
    manifest=dict(proof_sha256=sha(BASE/'verified_report.json'),plot_source_sha256=sha(Path(__file__)),
        artifacts_sha256={n:sha(BASE/n) for n in names[:3]},limits=report['limits'])
    (BASE/names[3]).write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
