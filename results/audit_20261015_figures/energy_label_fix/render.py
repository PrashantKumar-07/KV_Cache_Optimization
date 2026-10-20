from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import plot_figures_v2 as style
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
BASE=Path(__file__).resolve().parent
original=Figure.savefig
paths=[]
def save(self,path,*args,**kw):
 self.axes[0].set_ylabel('Modeled decode KV-traffic energy (J)')
 self.axes[0].set_title('HISTORICAL: analytical decode KV-traffic energy at assumed prices',fontsize=11)
 self.texts[0].set_text('Unequal output lengths; assumed STT 2/8 and VRAM 12 pJ/bit. Residual is other billed traffic, not isolated attention.')
 handles,labels=self.axes[0].get_legend_handles_labels();labels[0]='Other billed\ntraffic (residual)'
 self.axes[0].legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.15))
 target=BASE/Path(path).name;original(self,target,*args,**kw);paths.append(target)
Figure.savefig=save
source=ROOT/'results/remediation_v3/equal_vram_merged.json';style.fig12_energy(json.loads(source.read_text()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(BASE/'provenance.json').write_text(json.dumps({'purpose':'Prevent long yaxislabel clipping; preserve all bars/colors/assumptions; history only','input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [source,ROOT/'src/cost_model.py',ROOT/'experiments/plot_figures_v2.py',Path(__file__)]},'outputs':{p.name:sha(p) for p in paths}},indent=2)+'\n')
