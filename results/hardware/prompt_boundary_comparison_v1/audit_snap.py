"""Fresh faithful SnapKV allocation envelope with native last-prefill logits."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralConfig, MistralForCausalLM
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments'))
from audit_snap_tier_resources import snapshot
from longbench_eval import _attn_impl, prefill_logit_kwargs
from measure_fresh_process_memory import PinnedRoots, memory
from provenance import atomic_json, runtime_identity
from snapkv_adapter import SnapKVAdapter


def hooks(model):
    return [(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]


@torch.inference_mode()
def run(model,ids,stream,*,last=True,capacity=682,window=32,observer=None):
    baseline=hooks(model);groups=[layer.self_attn.num_key_value_groups for layer in model.model.layers]
    cache=SnapKVAdapter(model,capacity=capacity,window=window,kernel=5)
    out=past=None;predictions=[];widths=[]
    try:
        with _attn_impl(model,'sdpa'):
            out=model(ids,past_key_values=cache,use_cache=True,
                position_ids=torch.arange(ids.shape[1],device=ids.device)[None],
                **prefill_logit_kwargs(model,last))
        past=out.past_key_values
        prefill_token=out.logits[:,-1:].argmax(-1).item()
        if observer:observer('prefill_compressed',past,None)
        del out
        for step,teacher in enumerate(stream):
            before=[layer.keys.shape[-2]+1 for layer in past.layers]
            widths.append(before)
            out=model(torch.tensor([[teacher]],device=ids.device),past_key_values=past,use_cache=True,
                position_ids=torch.tensor([[ids.shape[1]+step]],device=ids.device),output_attentions=False)
            past=out.past_key_values
            predictions.append(out.logits[:,-1:].argmax(-1).item())
            if observer:observer(f'decode_{step+1}',past,before)
        return dict(prefill_prediction=prefill_token,predicted_token_ids=predictions,
                    forward_attended_rows_by_layer=widths)
    finally:
        cache.close();past=out=None
        assert hooks(model)==baseline,'Adapter hook leak'
        assert [layer.self_attn.num_key_value_groups for layer in model.model.layers]==groups


def cache_hash(past):
    digest=hashlib.sha256()
    for layer in past.layers:
        for value in (layer.keys,layer.values):
            digest.update(str((tuple(value.shape),value.dtype)).encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check_cpu():
    path=Path(__file__).with_name('cpu.json');assert not path.exists()
    torch.set_num_threads(2);torch.manual_seed(0)
    model=MistralForCausalLM(MistralConfig(hidden_size=32,intermediate_size=64,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,
        vocab_size=64,sliding_window=None,attn_implementation='eager')).eval()
    pairs=[]
    for seq,cap in ((8,12),(32,12),(64,24)):
        ids=(torch.arange(seq)%60+1)[None];arms=[]
        for last in (False,True):
            states=[]
            observer=lambda stage,past,width:states.append(dict(stage=stage,state_sha256=cache_hash(past),width=width))
            arms.append(dict(**run(model,ids,[1,2,3,4,5],last=last,capacity=cap,window=4,observer=observer),states=states))
        assert arms[0]==arms[1],'Native last-logit predictions/cache/width changed'
        assert arms[0]['forward_attended_rows_by_layer']==[[min(seq,cap)+i+1]*3 for i in range(5)]
        pairs.append(dict(seq=seq,capacity=cap,exact=True))
    # An attention failure after adapter installation must restore all hooks/groups.
    baseline=hooks(model);groups=[layer.self_attn.num_key_value_groups for layer in model.model.layers]
    def fail(*args):raise RuntimeError('deliberate attention failure')
    handle=model.model.layers[1].register_forward_pre_hook(fail)
    try:
        try:run(model,torch.arange(1,17)[None],[1],capacity=12,window=4)
        except RuntimeError as e:assert str(e)=='deliberate attention failure'
        else:raise AssertionError('Expected injected failure')
    finally:handle.remove()
    assert hooks(model)==baseline
    assert [layer.self_attn.num_key_value_groups for layer in model.model.layers]==groups
    atomic_json(path,dict(passed=True,pairs=pairs,partial_error_cleanup=True))
    print('Three CPU native-all/last pairs and partial-error cleanup pass',flush=True)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-check',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.cpu_check:return check_cpu()
    assert args.output is not None and not args.output.exists()
    plan=json.loads(Path(__file__).with_name('plan.json').read_text())
    original_empty=torch.empty;tracker=PinnedRoots();report=dict(passed=False,runtime=runtime_identity(),boundaries=[])
    model=None;baseline=None
    try:
        torch.manual_seed(plan['seed'])
        model=AutoModelForCausalLM.from_pretrained(plan['model'],torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(plan['model'])
        prompt=Path('results/remediation_v3/latency_prompt.txt').read_text()
        teacher_path=Path('results/shadow_ablation_full/fixed_stream_20260929.json')
        teachers=json.loads(teacher_path.read_text())['config']['teacher_token_ids'][:50]
        ids=tokenizer(prompt,truncation=False,return_tensors='pt').input_ids[:,:16384].to('cuda')
        assert ids.shape==(1,16384) and len(teachers)==50
        baseline=hooks(model)
        roots={t.untyped_storage().data_ptr():t.untyped_storage().nbytes()
               for t in list(model.parameters())+list(model.buffers()) if t.device.type=='cuda'}
        report.update(model=plan['model'],seed=plan['seed'],context=16384,decode=50,
            capacity=682,window=32,kernel=5,prefill_last_logits=True,gpu=torch.cuda.get_device_name(),
            constant_model_storage_bytes=sum(roots.values()),teacher_token_ids=teachers,
            input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),teacher_source_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
            source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in plan['control_sources_sha256']},
            audit_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        report['loaded_model']=memory('loaded_model')
        torch.empty=tracker.empty
        phase=dict(whole_allocated_peak=0,whole_reserved_peak=0,prefill_peak=0,decode_peak=0,decode_reserved_peak=0)
        def observe(stage,past,width):
            torch.cuda.synchronize()
            row=memory(stage);row['cache']=snapshot(past,None,len(report['boundaries']))
            row['forward_attended_rows_by_layer']=width
            phase['whole_allocated_peak']=max(phase['whole_allocated_peak'],row['gpu_peak_allocated_bytes'])
            phase['whole_reserved_peak']=max(phase['whole_reserved_peak'],row['gpu_peak_reserved_bytes'])
            if stage=='prefill_compressed':
                phase['prefill_peak']=row['gpu_peak_allocated_bytes'];torch.cuda.reset_peak_memory_stats()
            else:
                phase['decode_peak']=max(phase['decode_peak'],row['gpu_peak_allocated_bytes'])
                phase['decode_reserved_peak']=max(phase['decode_reserved_peak'],row['gpu_peak_reserved_bytes'])
            report['boundaries'].append(row)
        report.update(run(model,ids,teachers,observer=observe))
        gc.collect();torch.cuda.synchronize()
        assert tracker.current==0 and tracker.peak==0,'Unexpected owned pinned storage'
        assert hooks(model)==baseline
        report.update(passed=True,gpu_phase_peaks=phase,owned_pinned_peak_bytes=tracker.peak,
            owned_pinned_after_cleanup_bytes=tracker.current,adapter_hooks_and_groups_restored=True,
            explicit_h2d_kv_bytes=0,explicit_d2h_kv_bytes=0,
            limitations='Fresh one-model16K50 faithful expanded-head/growing-cache audit. Phase counters match ordinary handoff scope. '
                'Primitive boundary observations invalidate times. Owned requested pinned roots zero; no hidden retained allocator, '
                'driver allocation, exact-resource/width equality, joint quality, speed or physicalSTT claim.')
    except BaseException as e:
        report['error']=f'{type(e).__name__}: {e}';raise
    finally:
        torch.empty=original_empty
        atomic_json(args.output,report)
        if model is not None and baseline is not None:assert hooks(model)==baseline
    print('Snap actual envelope complete',report['gpu_phase_peaks'],flush=True)


if __name__=='__main__':main()
