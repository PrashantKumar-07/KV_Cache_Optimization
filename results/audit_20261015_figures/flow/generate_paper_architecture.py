"""Editable two-column system diagram: data path, control loop and shadow writes."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'tieredkv-architecture-v2','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(13.6,7.6));fig.patch.set_facecolor('white')
ax.set(xlim=(0,13.6),ylim=(0,7.6));ax.axis('off');fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
NAVY='#24364b';BLUE='#457b9d';TEAL='#287b70';RED='#b65243';GRAY='#657383';LIGHT='#f3f6f8'
def label(x,y,s,fs=12,color=NAVY,weight='normal',ha='left',va='center'):
 return ax.text(x,y,s,fontsize=fs,color=color,fontweight=weight,ha=ha,va=va,linespacing=1.45)
def box(x,y,w,h,title,number,color=BLUE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=.12',facecolor='white',edgecolor=color,linewidth=1.5))
 ax.add_patch(FancyBboxPatch((x,y+h-.52),w,.52,boxstyle='round,pad=0.025,rounding_size=.12',facecolor=color,edgecolor=color))
 label(x+.15,y+h-.25,f'{number}  {title}',13,'white','bold')
def arrow(points,color=NAVY,dashed=False):
 for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=color,linewidth=1.6,linestyle='--' if dashed else '-')
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=14,linewidth=1.6,color=color,linestyle='--' if dashed else '-'))
label(.35,7.18,'TieredKV: data movement and write-aware cache control',21,weight='bold')
label(.35,6.77,'Native prototype shown; two retained tiers, with immutable cold backups.',12,color=GRAY)
# Initialization and hot cache remain distinct from the recurrent model call.
box(.35,4.4,2.65,1.95,'Prefill',1,GRAY)
label(.55,5.47,'Full prompt → model KV',12)
label(.55,5.05,'Observation-window ranking',11.5)
label(.55,4.68,'Split hot / live cold / drop',11.5)
box(3.55,4.4,3.2,1.95,'GPU hot working set',2,BLUE)
for r in range(2):
 for c in range(8):
  color=BLUE if c<2 else TEAL if c>5 else '#b6c3cf'
  ax.add_patch(Rectangle((3.78+c*.345,5.1-r*.35),.28,.25,facecolor=color,edgecolor='white',linewidth=.8))
label(3.78,4.65,'Sinks + recent + retained rows',11.2)
box(7.5,4.4,3.0,1.95,'Model attention',3,BLUE)
label(7.7,5.44,'Hot + selected live cold',12)
label(7.7,5.03,'+ current-token KV',12)
label(7.7,4.65,'Exact attention on this set',11.3)
ax.add_patch(FancyBboxPatch((11.35,4.88),1.85,.98,boxstyle='round,pad=.025,rounding_size=.12',facecolor=LIGHT,edgecolor=NAVY,linewidth=1.3))
label(12.275,5.51,'Next token',13,weight='bold',ha='center')
label(12.275,5.17,'logits / argmax',10.5,ha='center')
arrow([(3.0,5.55),(3.55,5.55)])
arrow([(6.75,5.55),(7.5,5.55)])
arrow([(10.5,5.38),(11.35,5.38)])
# Live payload and shadow identity are concrete state, not a third memory tier.
box(.35,1.8,6.4,2.03,'Pinned CPU cold tier', 'KV',TEAL)
label(.55,3.03,'Live rows: candidates for quota-limited exposure',11.5)
for c in range(9):
 ax.add_patch(Rectangle((.58+c*.59,2.52),.48,.27,facecolor='#b7dbd3' if c<6 else 'white',edgecolor=TEAL,linewidth=1.1,hatch=None if c<6 else '//'))
label(.57,2.23,'Live KV',10.5,color=TEAL)
label(4.05,2.23,'Shadow = copy of hot KV',10.5,color=TEAL)
arrow([(1.3,4.4),(1.3,3.83)],color=GRAY)
label(1.43,4.08,'initial cold rows',10,color=GRAY)
# Cold exposure joins the attention input, not the policy's post-forward scores.
arrow([(6.75,3.04),(7.12,3.04),(7.12,4.86),(7.5,4.86)],color=TEAL)
label(6.93,4.04,'Live-row\nselection',10,color=TEAL,ha='right')
box(7.5,1.8,5.7,2.03,'Feedback & cache update',4,RED)
label(7.72,3.03,'Update attention scores; promote selected cold rows',11.4)
label(7.72,2.65,'Protect sinks/recent; evict low age-normalized scores',11.1)
label(7.72,2.25,'Prepare the next visible set; drop on cold overflow',11.1)
arrow([(9,4.4),(9,3.83)],color=RED)
label(9.15,4.1,'attention feedback',10.5,color=RED)
arrow([(7.5,2.62),(6.75,2.62)],color=RED)
label(7.125,2.92,'update',9.5,color=RED,ha='center')
arrow([(10.95,3.83),(10.95,6.62),(5.15,6.62),(5.15,6.35)],color=RED,dashed=True)
label(9.35,6.68,'promote / compact hot state for next step',10.5,color=RED,ha='center')
# The actual contribution is represented as a write decision with explicit bounds.
ax.add_patch(FancyBboxPatch((.35,.66),12.85,.76,boxstyle='round,pad=.025,rounding_size=.09',facecolor=LIGHT,edgecolor='#d2d9df',linewidth=1))
label(.58,1.13,'Hot victim already has a shadow?',12,weight='bold')
label(5.08,1.13,'YES  Reactivate; no KV payload write',11.5,color=TEAL)
label(9.6,1.13,'NO  Append cold KV',11.5,color=RED)
label(.58,.86,'No third storage tier: excess cold rows are permanently dropped.',10.5,color=GRAY)
label(.35,.28,'STT-RAM latency/energy are analytical assumptions; this implementation uses CPU DRAM. Default functional mode stores cold KV on GPU.',9.2,color=GRAY)
outputs=[]
for ext in ['svg','pdf','png']:
 p=OUT/('tieredkv_architecture_paper.'+ext)
 fig.savefig(p,dpi=240,metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
 outputs.append(p)
plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=['experiments/longbench_eval.py','experiments/cold_boundary_policy.py','experiments/cold_boundary.py','experiments/reusable_exact_root.py','src/tiered_kv_cache.py']
(OUT/'paper_architecture_provenance.json').write_text(json.dumps({'purpose':'Userauthorizedredesign: compactpaperarchitecture withdatapath/control+shadowwritebehavior, not priorgiantflowposter','generator_sha256':sha(Path(__file__)),'source_sha256':{s:sha(ROOT/s) for s in sources},'outputs_sha256':{p.name:sha(p) for p in outputs},'limits':'NativeCPUprototype visualized; pureGPUfunctional alternative andanalyticalSTT explicit. Schematiccellcountsnotexperimentalcapacity. Exactattention onlyoverpreparedvisible set, notfullcache.'},indent=2)+'\n')
