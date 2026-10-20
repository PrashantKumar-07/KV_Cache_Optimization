"""CPU-only feasibility evidence; does not implement native query selection."""
from pathlib import Path
import hashlib,json,sys
import torch
from transformers import MistralConfig,MistralForCausalLM
from transformers.models.mistral.modeling_mistral import rotate_half
from transformers.cache_utils import DynamicLayer
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from reserved_decode_append import append_reserved,capacity

torch.set_num_threads(1);torch.manual_seed(15)
model=MistralForCausalLM(MistralConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
    num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
    max_position_embeddings=64,sliding_window=None)).to(torch.bfloat16).eval()
contexts={};captured={};events=[];handles=[]
for i,layer in enumerate(model.model.layers):
    def before(module,args,kwargs,i=i):
        contexts[i]=(kwargs.get('hidden_states',args[0] if args else None),kwargs['position_embeddings'])
        events.append([i,'attention_before'])
    def projected(module,args,out,i=i):
        hidden,(cos,sin)=contexts[i]
        duplicate=torch.nn.functional.linear(hidden,module.weight,module.bias)
        assert torch.equal(out,duplicate)
        q=out.view(1,hidden.shape[1],4,8).transpose(1,2)
        rotary=q*cos.unsqueeze(1)+rotate_half(q)*sin.unsqueeze(1)
        captured[i]=rotary.detach().clone();events.append([i,'actual_q_projection'])
    handles.append(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True))
    handles.append(layer.self_attn.q_proj.register_forward_hook(projected))
with torch.no_grad():
    first=model(torch.tensor([[1,2,3,4]]),use_cache=True)
    past=first.past_key_values
    for i,layer in enumerate(past.layers):
        original=layer.update
        def update(key,value,*args,original=original,i=i,**kwargs):
            assert captured[i].shape==(1,4,1,8)
            events.append([i,'cache_append']);return original(key,value,*args,**kwargs)
        layer.update=update
    events.clear()
    model(torch.tensor([[5]]),past_key_values=past,position_ids=torch.tensor([[4]]),use_cache=True)
for h in handles:h.remove()
assert events==[[i,name] for i in range(2) for name in ['attention_before','actual_q_projection','cache_append']]

with torch.no_grad():
    layer=DynamicLayer()
    layer.update(torch.zeros(1,2,6,8,dtype=torch.bfloat16),torch.zeros(1,2,6,8,dtype=torch.bfloat16))
    buffers=[torch.zeros(1,2,7,8,dtype=torch.bfloat16) for _ in range(2)]
    layer.keys,layer.values=[b[:,:,:6] for b in buffers]
    roots=[t.untyped_storage()._cdata for t in [layer.keys,layer.values]]
    # Simulated in-place cold-tail replacement, hot4+cold2.
    layer.keys[:,:,4:6].copy_(torch.ones(1,2,2,8,dtype=torch.bfloat16))
    layer.values[:,:,4:6].copy_(torch.full((1,2,2,8),2,dtype=torch.bfloat16))
    assert [t.untyped_storage()._cdata for t in [layer.keys,layer.values]]==roots
    new=torch.full((1,2,1,8),3,dtype=torch.bfloat16)
    assert append_reserved(layer,new,new) is not None
    assert torch.equal(layer.keys[:,:,:4],torch.zeros(1,2,4,8,dtype=torch.bfloat16))
    assert torch.equal(layer.keys[:,:,4:6],torch.ones(1,2,2,8,dtype=torch.bfloat16))
    assert torch.equal(layer.keys[:,:,6:7],new)
    # Dense-concat replacement loses one-row spare and ownership.
    layer.keys=torch.cat([buffers[0][:,:,:4],torch.ones(1,2,2,8,dtype=torch.bfloat16)],dim=2)
    layer.values=torch.cat([buffers[1][:,:,:4],torch.ones(1,2,2,8,dtype=torch.bfloat16)],dim=2)
    assert capacity(layer.keys)==6
    assert append_reserved(layer,new,new) is None
out={'CPU_only':True,'torch':torch.__version__,'seed':15,'all_checks_passed':True,
 'two_layer_decode_event_order':events,'actual_projection_equals_duplicate':True,
 'in_place_tail_keeps_root_and_reserved_append':True,'dense_concat_forces_reserved_append_fallback':True,
 'limitation':'Tiny CPU Mistral supports event ordering and buffer mechanics; no native CPU-controller/query-selector implementation or GPU performance proof',
 'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
  ['experiments/evaluate_query_selection.py','experiments/cpu_phase_policy.py','experiments/cold_boundary.py','experiments/reserved_decode_append.py']}}
print(json.dumps(out,indent=2))
