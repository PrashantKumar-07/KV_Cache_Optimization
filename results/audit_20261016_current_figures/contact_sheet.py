"""Review-only overview of the nine native scientific plots."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
names=[name for name in json.loads((BASE/'provenance.json').read_text())['outputs_sha256'] if name.endswith('.png')]
assert len(names)==9
fig,axes=plt.subplots(3,3,figsize=(18,14))
for ax,name in zip(axes.flat,names):
    ax.imshow(plt.imread(BASE/name));ax.axis('off');ax.set_title(name,fontsize=9)
fig.tight_layout(pad=1)
fig.savefig(BASE/'contact_sheet.png',dpi=180)
plt.close(fig)
