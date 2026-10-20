"""Partial-forward errors restore adapter hooks/groups and release root owners."""
import gc
import json
from pathlib import Path
import sys
import torch
from transformers import MistralConfig,MistralForCausalLM
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from candidate import ReservedSnapKV
from longbench_eval import _attn_impl


def hooks(model):return [(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]


def main():
    path=Path(__file__).with_name('errors.json');assert not path.exists()
    torch.set_num_threads(2);torch.manual_seed(0)
    model=MistralForCausalLM(MistralConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=64,sliding_window=None,
        attn_implementation='eager')).eval()
    rows=[]
    for phase in ('prefill','decode'):
        baseline=hooks(model);groups=[l.self_attn.num_key_value_groups for l in model.model.layers]
        refs=[];cache=ReservedSnapKV(model,capacity=12,window=4,max_rows=17);out=None
        def fail(*args):raise RuntimeError('deliberate partial '+phase)
        handle=None
        try:
            if phase=='prefill':handle=model.model.layers[1].register_forward_pre_hook(fail)
            with torch.inference_mode(),_attn_impl(model,'sdpa'):
                if phase=='prefill':
                    try:out=model(torch.arange(1,33)[None],past_key_values=cache,use_cache=True,logits_to_keep=1)
                    except RuntimeError as e:assert str(e)=='deliberate partial prefill'
                    else:raise AssertionError('Expected injected error')
                else:
                    out=model(torch.arange(1,33)[None],past_key_values=cache,use_cache=True,logits_to_keep=1)
                    handle=model.model.layers[1].register_forward_pre_hook(fail)
                    try:out=model(torch.tensor([[1]]),past_key_values=cache,use_cache=True,position_ids=torch.tensor([[32]]))
                    except RuntimeError as e:assert str(e)=='deliberate partial decode'
                    else:raise AssertionError('Expected injected error')
            refs=[t.untyped_storage()._weak_ref() for pair in cache.roots.values() for t in pair]
            assert len(refs)==(2 if phase=='prefill' else 6)
        finally:
            if handle:handle.remove()
            cache.close();out=None;del cache;gc.collect()
            assert hooks(model)==baseline
            assert [l.self_attn.num_key_value_groups for l in model.model.layers]==groups
            assert all(torch.UntypedStorage._expired(ref) for ref in refs)
            for ref in refs:torch.UntypedStorage._free_weak_ref(ref)
        rows.append(dict(phase=phase,roots_released=len(refs),hooks_groups_restored=True))
    path.write_text(json.dumps(dict(passed=True,rows=rows),indent=2)+'\n')
    print('Partial prefill/decode failures release all roots and restore hooks/groups',flush=True)


if __name__=='__main__':main()
