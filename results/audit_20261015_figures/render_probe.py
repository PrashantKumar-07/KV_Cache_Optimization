"""Regenerate only audit copies, recording every plotted artist's numeric data."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
MODE=sys.argv[1]
OUT=Path(__file__).resolve().parent/'reproductions'/MODE
OUT.mkdir(parents=True,exist_ok=False)
original=Figure.savefig
records=[]
def plain(value):
 if hasattr(value,'tolist'):return value.tolist()
 if isinstance(value,(np.integer,np.floating)):return value.item()
 return value
def save(self,fname,*args,**kwargs):
 name=Path(fname).name
 if name.endswith('.png'):
  axes=[]
  for ax in self.axes:
   axes.append({'title':ax.get_title(),'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),
    'ylim':list(ax.get_ylim()),'xticklabels':[x.get_text() for x in ax.get_xticklabels()],
    'bars':[{'x':p.get_x(),'y':p.get_y(),'width':p.get_width(),'height':p.get_height()} for p in ax.patches if hasattr(p,'get_height')],
    'lines':[{'label':l.get_label(),'x':plain(l.get_xdata()),'y':plain(l.get_ydata())} for l in ax.lines],
    'images':[plain(x.get_array()) for x in ax.images],
    'texts':[t.get_text() for t in ax.texts]})
  records.append({'figure':name,'axes':axes,'figure_texts':[t.get_text() for t in self.texts]})
 return original(self,OUT/name,*args,**kwargs)
Figure.savefig=save
load=lambda p:json.loads(Path(p).read_text())
if MODE=='v2_remediation_v3':
 import plot_figures_v2 as m
 sys.argv=['plot_figures_v2','--equal-vram','results/remediation_v3/equal_vram_merged.json','--fair','results/remediation_v3/equal_attended_merged.json','--latency','results/remediation_v3/latency.json','--outdir',str(OUT),'--tex-outdir',str(OUT/'tables')]
 m.main()
elif MODE.startswith('publication_'):
 import plot_publication_figures as m
 if MODE.endswith('legacy') or MODE=='publication_legacy_title':
  ev=load('results/final_equal_vram_frac075.json');fair=load('results/final_equal_attended_frac075.json')
 else:
  ev=load('results/remediation_v3/equal_vram_merged.json');fair=load('results/remediation_v3/equal_attended_merged.json')
 
 if MODE=='publication_legacy_title':
  from matplotlib.axes import Axes
  old_title=Axes.set_title
  def historical_title(self,label,*args,**kwargs):
   return old_title(self,label.replace('(Matched Attention)','(Fair / Iso-VRAM)'),*args,**kwargs)
  Axes.set_title=historical_title
 m.fig2_latency(fair,str(OUT));m.fig3_write_savings(fair,ev,list(Path('results/clean_sweep_stt').glob('stt*.json')),str(OUT));m.fig6_migration_profile(ev,str(OUT))
elif MODE=='frac_compare':
 import plot_frac_compare as m
 m.main()
elif MODE=='frac_ablation':
 import plot_frac_ablation as m
 # Avoid the script's unrelated legacy table mutation; render the figure only.
 source=Path(m.__file__).read_text();start=source.index('def main():');end=source.index('    os.makedirs("results/tables"',start)
 body=source[start:end]+'\n'
 ns=m.__dict__;exec(body,ns);ns['main']()
for r in records:
 p=OUT/r['figure'];reference=ROOT/'figures_final'/r['figure']
 if reference.exists():
  a=np.asarray(Image.open(p));b=np.asarray(Image.open(reference))
  r['reference_sha256']=hashlib.sha256(reference.read_bytes()).hexdigest();r['regenerated_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  r['same_shape']=a.shape==b.shape;r['exact_pixels']=a.shape==b.shape and bool(np.array_equal(a,b));r['shape']=list(a.shape);r['reference_shape']=list(b.shape)
  if a.shape==b.shape:r['different_pixel_fraction']=float(np.mean(np.any(a!=b,axis=-1)))
(OUT/'artists.json').write_text(json.dumps(records,indent=2,default=plain))
print([(r['figure'],r.get('exact_pixels')) for r in records])
