"""Current original-five-method context scaling; verified quiet measurements."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(ROOT/'experiments'))
import plot_figures_v2 as style
import matplotlib.pyplot as plt
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof=json.loads((BASE/'verified_timing.json').read_text())
raw=json.loads((BASE/'timing.json').read_text())
plan=json.loads((BASE/'plan.json').read_text())
assert proof['verified'] and proof['execution_complete'] and raw['execution_complete']
assert sha(BASE/'timing.json')==proof['raw_sha256']
assert sha(BASE/'plan.json')==proof['plan_sha256']==raw['plan_sha256']
assert len(raw['rows'])==90 and len(raw['quiet_observations'])==181
contexts=[512,1024,2048,4096,8192,16384]
arms=['full','streamingllm','h2o_adapted','snapkv_adapted','tiered_hot1024']
labels=['Full Cache','StreamingLLM (adapted)','H2O (adapted)','SnapKV (adapted)','TieredKV']
keys=['full','streamingllm','h2o','snapkv','tieredkv']
assert plan['contexts']==contexts and plan['arms']==arms
rows=[]
for arm in arms:
    for ctx in contexts:
        samples=[r for r in raw['rows'] if r['arm']==arm and r['context']==ctx]
        assert len(samples)==3 and {r['block'] for r in samples}=={0,1,2}
        values=[v for r in samples for trial in r['trials'] for v in trial['step_ms']]
        assert len(values)==150 and all(math.isfinite(v) and v>0 for v in values)
        median=statistics.median(values)
        assert median==proof['summaries'][str(ctx)][arm]['median_itl_ms']
        rows.append({'arm':arm,'context_tokens':ctx,'measured_steps':len(values),'median_itl_ms':median})
assert sum(r['measured_steps'] for r in rows)==4500
with (HERE/'values.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
fig,ax=plt.subplots(figsize=(9.5,4.8))
for arm,label,key in zip(arms,labels,keys):
    values=[r['median_itl_ms'] for r in rows if r['arm']==arm]
    # Equal/near-equal H2O and Snap traces stay at their measured locations.
    # An open square exposes the H2O outline behind Snap's circle.
    ax.plot(contexts,values,'s-' if arm=='h2o_adapted' else 'o-',label=label,
            markerfacecolor='none' if arm=='h2o_adapted' else style.PALETTE[key],
            markersize=8 if arm=='h2o_adapted' else 7,color=style.PALETTE[key])
ax.set_xscale('log');ax.set_xticks(contexts,[f'{v:,}' for v in contexts]);ax.minorticks_off()
ax.set_xlabel('Context length (tokens)');ax.set_ylabel('Median full-loop decode latency (ms/token)')
ax.set_title('Decode latency including required feedback and cache policy',fontweight='bold',fontsize=11)
ax.set_ylim(0,max(r['median_itl_ms'] for r in rows)*1.12)
ax.legend(loc='center',bbox_to_anchor=(.70,.61),fontsize=8.5);style.despine(ax)
caption=('Mistral-7B / L40S; 3 blocks × 50 fixed-input steps per arm/context. Medians; no confidence intervals.\n'
         'SDPA: Full/Streaming. Eager feedback: adapted H2O/Snap/Tiered. SDPA prefill; common finite-check overhead included.\n'
         'Tiered: GPU simulation, 1,024 hot + 2,048 cold, 32 quota; extra capacity/attention. Not native-CPU or equal-memory/SOTA timing.')
fig.text(.5,.01,caption,ha='center',va='bottom',fontsize=7.2,color='#555555',linespacing=1.25)
fig.tight_layout(rect=(0,.16,1,1));fig.canvas.draw()
artists=[{'arm':arm,'x':list(map(float,line.get_xdata())),'y':list(map(float,line.get_ydata()))}
         for arm,line in zip(arms,ax.lines)]
for artist,arm in zip(artists,arms):
    assert artist['x']==contexts
    assert artist['y']==[proof['summaries'][str(ctx)][arm]['median_itl_ms'] for ctx in contexts]
outputs=[]
for ext in ('png','pdf'):
    p=HERE/f'fig4_decode_scaling.{ext}'
    fig.savefig(p,metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {});outputs.append(p)
plt.close(fig)
contract={'scope':proof['scope'],'caption':caption,'contexts':contexts,'arms':arms,'values':rows,'artists':artists,
    'backend_contract':proof['backend_contract'],'resource_scope':proof['resource_scope'],
    'raw_sha256':sha(BASE/'timing.json'),'verified_proof_sha256':sha(BASE/'verified_timing.json'),
    'no_uncertainty_intervals':True,'H2O_Snap_overlap':'Open H2O square distinguishes near-identical traces without moving values.'}
(HERE/'numeric_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
manifest={'status':'staged for root review; not deployed','generator_sha256':sha(__file__),
    'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [BASE/'timing.json',BASE/'verified_timing.json',BASE/'plan.json',ROOT/'experiments/plot_figures_v2.py']},
    'csv_sha256':sha(HERE/'values.csv'),'numeric_contract_sha256':sha(HERE/'numeric_contract.json'),
    'outputs_sha256':{p.name:sha(p) for p in outputs},'original_five_method_question_preserved':True,
    'measured_forwards':4500,'quiet_observations':181,'physical_backends_disclosed':True,'not_native_deployment_or_SOTA':True}
(HERE/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'staged':list(manifest['outputs_sha256']),'medians_checked':30}))
