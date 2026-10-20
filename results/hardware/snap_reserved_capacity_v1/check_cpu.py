"""CPU selector fidelity, prefix bits, genuine reservation and overflow guard."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch
from transformers import MistralConfig,MistralForCausalLM
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from candidate import ReservedSnapKV
from snapkv_adapter import SnapKVAdapter
from longbench_eval import _attn_impl


def payload(cache):
    return [(l.keys.clone(),l.values.clone()) for l in cache.layers]


def equal(a,b):
    return all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for p,q in zip(a,b) for x,y in zip(p,q))


def hooks(model):return [(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]


@torch.inference_mode()
def run(model,ids,cap,maximum,cls):
    before=hooks(model);groups=[l.self_attn.num_key_value_groups for l in model.model.layers]
    cache=cls(model,capacity=cap,window=4,**({'max_rows':maximum} if cls is ReservedSnapKV else {}))
    frames=[];tokens=[];reserved=[]
    try:
        with _attn_impl(model,'sdpa'):
            out=model(ids,past_key_values=cache,use_cache=True,position_ids=torch.arange(ids.shape[1])[None],logits_to_keep=1)
        frames.append(payload(cache));tokens.append(out.logits[:,-1:].argmax(-1).item())
        for step in range(5):
            out=model(torch.tensor([[step+1]]),past_key_values=cache,use_cache=True,
                position_ids=torch.tensor([[ids.shape[1]+step]]))
            frames.append(payload(cache));tokens.append(out.logits[:,-1:].argmax(-1).item())
            if cls is ReservedSnapKV:
                reserved.append(sum(t.untyped_storage().nbytes() for pair in cache.roots.values() for t in pair))
        if cls is ReservedSnapKV:
            expected=len(model.model.layers)*2*model.config.num_attention_heads*maximum*model.config.head_dim*next(model.parameters()).element_size()
            assert reserved==[expected]*5
            old=payload(cache)
            # Fill to the exact bound; one-over update must not write.
            layer=cache.layers[0]
            blank=torch.zeros(1,model.config.num_key_value_heads,1,model.config.head_dim,dtype=next(model.parameters()).dtype)
            while layer.keys.shape[-2]<maximum:cache.update(blank,blank,0)
            full=(layer.keys.clone(),layer.values.clone())
            try:cache.update(blank,blank,0)
            except ValueError:pass
            else:raise AssertionError('Expected reserved capacity error')
            assert equal([full],[(layer.keys,layer.values)])
        return tokens,frames,reserved
    finally:
        cache.close()
        assert hooks(model)==before
        assert [l.self_attn.num_key_value_groups for l in model.model.layers]==groups
        if cls is ReservedSnapKV:assert not cache.roots


def main():
    path=Path(__file__).with_name('cpu.json');assert not path.exists()
    torch.set_num_threads(2);rows=[]
    for dtype in (torch.float32,torch.bfloat16):
        torch.manual_seed(0)
        model=MistralForCausalLM(MistralConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,vocab_size=64,sliding_window=None,
            attn_implementation='eager')).to(dtype).eval()
        for seq,cap in ((1,12),(8,12),(32,12),(64,24)):
            ids=(torch.arange(seq)%60+1)[None]
            a=run(model,ids,cap,cap+5,SnapKVAdapter);b=run(model,ids,cap,cap+5,ReservedSnapKV)
            assert a[0]==b[0],'Predictions changed'
            assert all(equal(x,y) for x,y in zip(a[1],b[1])),'Populated KV bits changed'
            rows.append(dict(dtype=str(dtype),sequence=seq,capacity=cap,max_rows=cap+5,exact=True))
    # Constructor validation precedes any adapter hook installation.
    before=hooks(model)
    try:ReservedSnapKV(model,capacity=12,window=4,max_rows=11)
    except ValueError:pass
    else:raise AssertionError('Expected constructor capacity error')
    assert hooks(model)==before
    gc.collect()
    path.write_text(json.dumps(dict(passed=True,pairs=rows,populated_kv_bits_exact=True,
        overflow_before_mutation=True,constructor_before_hook_guard=True,all_hooks_groups_restored=True),indent=2)+'\n')
    print('Eight CPU pairs exact; actual backing/overflow/hook cleanup guards pass',flush=True)


if __name__=='__main__':main()
