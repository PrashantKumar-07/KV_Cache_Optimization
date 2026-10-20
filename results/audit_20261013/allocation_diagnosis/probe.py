"""Isolate pinned allocation overhead; no model/TTFT or STT timing claim."""
from pathlib import Path
import gc, hashlib, json, sys, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]; BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments'))
from exact_pinned_memory import allocate, stats
from audit_pinned_blocks import kernel
from provenance import atomic_json, runtime_identity
assert not (BASE/'observed.json').exists()
shape=(276824064,)
plan={'requested_bytes':shape[0],'order':['cached','exact','exact','cached']*3,
      'warmup_each':1,'scope':__doc__,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'runtime':runtime_identity()}
atomic_json(BASE/'plan.json',plan)
torch.cuda.set_device(0); anchor=torch.empty(1,device='cuda')
factories={'cached':lambda:torch.empty(shape,dtype=torch.uint8,pin_memory=True),
           'exact':lambda:allocate(shape,dtype=torch.uint8)}
for factory in factories.values():
    root=factory(); del root; gc.collect(); torch.cuda.synchronize()
rows=[]
for arm in plan['order']:
    torch.cuda.synchronize(); start=time.perf_counter()
    root=factories[arm](); allocation_ms=1000*(time.perf_counter()-start)
    info=kernel().block_info(root)
    start=time.perf_counter(); del root; gc.collect(); torch.cuda.synchronize()
    release_ms=1000*(time.perf_counter()-start)
    assert stats()['live_allocations']==0
    rows.append(dict(arm=arm,allocation_ms=allocation_ms,release_ms=release_ms,**info))
summary={arm:{name:float(np.median([r[name] for r in rows if r['arm']==arm]))
              for name in ['allocation_ms','release_ms']} for arm in factories}
atomic_json(BASE/'observed.json',dict(passed=True,plan=plan,rows=rows,summary=summary,
    limitations='Isolated serial allocation with warm CUDA context and cached Torch allocator; no model load, serving loop, overall TTFT, host RSS, or physical STT claim.'))
print(json.dumps(summary,indent=2))
