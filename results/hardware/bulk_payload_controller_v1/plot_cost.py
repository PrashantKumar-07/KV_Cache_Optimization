"""Export frozen component timing; no baseline/quality claim."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;F=B/'figures';F.mkdir(exist_ok=True)
p=B/'verified_cost.json';d=json.loads(p.read_text());assert d['verified'] and d['bulk_payload_controller_wholeloop_primary_passed']
rows=[]
for arm,label in [('control','CPU phase control'),('query','Bulk payload controller')]:
 for k in ['median_decode_ms','p95_decode_ms','median_ttft_ms']:rows.append(dict(arm=label,metric=k,value_ms=d['aggregate'][arm][k]))
with (F/'timing.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['arm','metric','value_ms']);w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(1,3,figsize=(9,3.1));labels=['CPU phase','Bulk payload']
for ax,(key,title) in zip(axes,[('median_decode_ms','Median decode'),('p95_decode_ms','p95 decode'),('median_ttft_ms','Median TTFT')]):
 vals=[d['aggregate'][arm][key] for arm in ['control','query']];bars=ax.bar(labels,vals,color=['#64748b','#16836b']);ax.set_title(title);ax.set_ylabel('Milliseconds');ax.set_ylim(0,max(vals)*1.22)
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+max(vals)*.025,f'{v:.2f}',ha='center',fontsize=9)
 ax.spines[['top','right']].set_visible(False)
fig.suptitle('Same-control component timing: Mistral-7B, 16K prompt, 50 decode steps',fontsize=10)
fig.text(.5,.01,'3 ABBA blocks; 300 measured forwards/arm. No faithful-baseline or independent quality claim.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.045,1,.94))
for ext in ['png','pdf']:fig.savefig(F/('timing.'+ext),dpi=180)
plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(F/'manifest.json').write_text(json.dumps(dict(input_sha256=sha(p),plot_sha256=sha(Path(__file__)),exports={q.name:sha(q) for q in F.iterdir() if q.suffix in ['.csv','.png','.pdf']},limits=d['limits']),indent=2)+'\n')
