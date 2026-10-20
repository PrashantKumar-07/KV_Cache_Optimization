"""Isolated CPU failure injection; original source and active jobs untouched."""
from pathlib import Path
import hashlib,json,sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from evaluate_query_selection import PromptQueryCapture
from reserved_decode_append import AppendLease
class Policy: pass

def fail(*args,**kw):raise RuntimeError('injected second registration failure')
attentions=[torch.nn.Linear(2,2) for _ in range(2)]
model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=a) for a in attentions]))
policy=Policy();policy.prefill_chunk_size=0
saved=[];original=attentions[0].register_forward_pre_hook

def register(*args,**kwargs):
 handle=original(*args,**kwargs);saved.append(handle);return handle
with patch.object(attentions[0],'register_forward_pre_hook',register),patch.object(attentions[1],'register_forward_pre_hook',fail):
 try:PromptQueryCapture(model,policy)
 except RuntimeError as error:assert str(error)=='injected second registration failure'
 else:raise AssertionError('fault did not trigger')
capture_leaked=[len(a._forward_pre_hooks) for a in attentions]
assert capture_leaked==[1,0]
for h in saved:h.remove()
assert all(not a._forward_pre_hooks for a in attentions)

model=torch.nn.Linear(2,2);policy=Policy();saved=[];original=model.register_forward_pre_hook

def register2(*args,**kwargs):
 handle=original(*args,**kwargs);saved.append(handle);return handle
with patch.object(model,'register_forward_pre_hook',register2),patch.object(model,'register_forward_hook',fail):
 try:AppendLease(model,policy)
 except RuntimeError as error:assert str(error)=='injected second registration failure'
 else:raise AssertionError('fault did not trigger')
append_leaked=(len(model._forward_pre_hooks),len(model._forward_hooks))
assert append_leaked==(1,0)
for h in saved:h.remove()
assert not model._forward_pre_hooks and not model._forward_hooks
out={'CPU_only':True,'reproduced':True,'prompt_capture_partial_registration_leaked_layer_hooks':capture_leaked,
 'append_lease_partial_registration_leaked_model_hooks':append_leaked,'isolated_probe_leaks_manually_closed':True,
 'locations':['experiments/evaluate_query_selection.py:130-151','experiments/reserved_decode_append.py:72-76'],
 'minimal_remedy':'Initialize handles list, guard registration try/except, remove every acquired handle then re-raise; outer ExitStack cannot close an object whose constructor failed.',
 'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
 ['experiments/evaluate_query_selection.py','experiments/reserved_decode_append.py']}}
print(json.dumps(out,indent=2))
