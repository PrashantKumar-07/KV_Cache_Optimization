"""Untimed full-Mistral every-populated-KV-bit and teacher-token preservation."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from candidate import ReservedSnapKV
from snapkv_adapter import SnapKVAdapter
from longbench_eval import _attn_impl
from provenance import atomic_json,runtime_identity


def hooks(model):return [(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]


def compare(a,b):
    assert len(a.layers)==len(b.layers)==32
    for left,right in zip(a.layers,b.layers):
        for x,y in ((left.keys,right.keys),(left.values,right.values)):
            assert x.shape==y.shape and x.dtype==y.dtype==torch.bfloat16
            assert torch.equal(x.view(torch.int16),y.view(torch.int16)),'Populated KV bits changed'
    return sum(t.numel()*t.element_size() for l in a.layers for t in (l.keys,l.values))


@torch.inference_mode()
def main():
    base=Path(__file__).resolve().parent;path=base/'gpu_preservation.json';assert not path.exists()
    report=dict(passed=False,runtime=runtime_identity(),frames=[]);a=b=None;refs=[]
    try:
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        prompt=Path('results/remediation_v3/latency_prompt.txt').read_text()
        teachers=json.loads(Path('results/shadow_ablation_full/fixed_stream_20260929.json').read_text())['config']['teacher_token_ids'][:50]
        ids=tokenizer(prompt,truncation=False,return_tensors='pt').input_ids[:,:16384].to('cuda')
        baseline=hooks(model);groups=[l.self_attn.num_key_value_groups for l in model.model.layers]
        report.update(input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),teacher_token_ids=teachers)
        a=SnapKVAdapter(model,capacity=974,window=32,kernel=5)
        with _attn_impl(model,'sdpa'):
            out=model(ids,past_key_values=a,use_cache=True,position_ids=torch.arange(16384,device='cuda')[None],logits_to_keep=1)
        logits=out.logits.clone();out=None
        # Close only the control's observation hooks; retain its populated cache.
        a.close();assert hooks(model)==baseline
        b=ReservedSnapKV(model,capacity=974,window=32,kernel=5,max_rows=1024)
        with _attn_impl(model,'sdpa'):
            out=model(ids,past_key_values=b,use_cache=True,position_ids=torch.arange(16384,device='cuda')[None],logits_to_keep=1)
        assert torch.equal(logits,out.logits),'Prefill logits changed'
        bits=compare(a,b);report['frames'].append(dict(step=0,rows=974,populated_bytes=bits,bits_exact=True))
        refs=[t.untyped_storage()._weak_ref() for pair in b.roots.values() for t in pair]
        assert sum(t.untyped_storage().nbytes() for pair in b.roots.values() for t in pair)==512*2**20
        predictions=[]
        for step,teacher in enumerate(teachers):
            token=torch.tensor([[teacher]],device='cuda');pos=torch.tensor([[16384+step]],device='cuda')
            out=model(token,past_key_values=a,use_cache=True,position_ids=pos)
            predicted=out.logits[:,-1:].argmax(-1).item();logits=out.logits.clone();out=None
            out=model(token,past_key_values=b,use_cache=True,position_ids=pos)
            assert predicted==out.logits[:,-1:].argmax(-1).item(),'Teacher prediction changed'
            max_delta=float((logits.float()-out.logits.float()).abs().max().item())
            bits=compare(a,b)
            assert b.layers[0].keys.shape[-2]==975+step
            predictions.append(predicted)
            report['frames'].append(dict(step=step+1,rows=975+step,populated_bytes=bits,bits_exact=True,
                prediction_exact=True,max_logit_absolute_delta=max_delta))
        blank=torch.zeros(1,8,1,128,dtype=torch.bfloat16,device='cuda')
        full=(b.layers[0].keys.clone(),b.layers[0].values.clone())
        try:b.update(blank,blank,0)
        except ValueError:pass
        else:raise AssertionError('Expected capacity error')
        assert all(torch.equal(x.view(torch.int16),y.view(torch.int16)) for x,y in zip(full,(b.layers[0].keys,b.layers[0].values)))
        full=None;b.close();a.close();out=logits=a=b=None;gc.collect();torch.cuda.synchronize()
        assert all(torch.UntypedStorage._expired(ref) for ref in refs),'Reserved root still owned'
        assert hooks(model)==baseline and [l.self_attn.num_key_value_groups for l in model.model.layers]==groups
        report.update(passed=True,predicted_token_ids=predictions,frames_verified=51,reserved_kv_bytes=512*2**20,
            roots_released=64,overflow_before_mutation=True,hooks_groups_restored=True,
            limitations='Untimed dual-cache preservation process; no resource peak or speed/quality superiority claim.')
    except BaseException as e:report['error']=f'{type(e).__name__}: {e}';raise
    finally:
        if b is not None:b.close()
        if a is not None:a.close()
        for ref in refs:torch.UntypedStorage._free_weak_ref(ref)
        atomic_json(path,report)
    print('GPU51 frames every populatedKV bit exact;50 predictions/overflow/root cleanup pass',flush=True)


if __name__=='__main__':main()
