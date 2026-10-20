"""Original GPU query math versus paired GPU/native CPU full-state pilot."""
import gc,hashlib,json,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from bulk_promotion_policy import BulkPromotionPolicy
from compare_cpu_batched import digest
from cpu_slot_cache import CpuSlotCache
from evaluate_query_selection import QuerySelector,PromptQueryCapture
from pool_query_selector import PoolQuerySelector
from longbench_eval import TieredKVPolicy,_attn_impl,to_tuple_kv,apply_tuple_kv_inplace
from provenance import atomic_json,runtime_identity


def hooks(model):return [(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in model.modules()]
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.inference_mode()
def run(model,ids,teachers,arm):
    policy=(BulkPromotionPolicy if arm=='cpu_paired' else TieredKVPolicy)(sram_budget=991,
        stt_budget=1090,stt_expose_quota=32,start_size=4,recent_size=128,
        kv_dtype=torch.bfloat16,sttram_bifurcation_frac=.75)
    before=hooks(model);capture=PromptQueryCapture(model,policy)
    selector=(QuerySelector if arm=='gpu_original' else PoolQuerySelector)(model,policy)
    observers=[];selections=[];active={'selector':False};canonical_reads=[0]
    original_canonical=CpuSlotCache._canonical
    def canonical(cache,*args,**options):
        if active['selector']:
            canonical_reads[0]+=1
            raise AssertionError('Native selector inspected canonical CPU cold tensor')
        return original_canonical(cache,*args,**options)
    CpuSlotCache._canonical=canonical
    if arm!='gpu_original':
        for handle in selector.handles:handle.remove()
        selector.handles.clear()
        for i,layer in enumerate(model.model.layers):
            select=selector._hook(i)
            def checked(module,args,options,select=select):
                active['selector']=True
                try:return select(module,args,options)
                finally:active['selector']=False
            selector.handles.append(layer.self_attn.register_forward_pre_hook(checked,with_kwargs=True))
    for i,layer in enumerate(model.model.layers):
        def observe(module,args,options,index=i):
            if index in policy.layer_caches and options['hidden_states'].shape[1]==1:
                past=options['past_key_values'];row=past.layers[index]
                positions=list(policy._vis_stt_pos[index]);indices=policy._vis_stt_idx[index]
                if arm!='gpu_original':assert positions==[policy.layer_caches[index].stt_pos[j] for j in indices]
                selections.append(dict(layer=index,positions=positions,cache_rows=row.keys.shape[-2],
                    visible_keys_sha256=tensor_hash(row.keys),visible_values_sha256=tensor_hash(row.values)))
        observers.append(layer.self_attn.register_forward_pre_hook(observe,with_kwargs=True))
    try:
        with _attn_impl(model,'sdpa'):
            out=model(ids,use_cache=True,position_ids=torch.arange(ids.shape[1],device='cuda')[None],logits_to_keep=1)
        past=out.past_key_values;apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
        capture.close();out=None
        frames=[dict(step=0,state_sha256=digest(policy.layer_caches),counters=policy._live_totals())]
        predictions=[]
        for step,teacher in enumerate(teachers):
            out=model(torch.tensor([[teacher]],device='cuda'),past_key_values=past,use_cache=True,
                position_ids=torch.tensor([[ids.shape[1]+step]],device='cuda'),output_attentions=True)
            past=out.past_key_values;policy.update_scores(to_tuple_kv(past),out.attentions)
            apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
            predictions.append(out.logits[:,-1:].argmax(-1).item())
            frames.append(dict(step=step+1,state_sha256=digest(policy.layer_caches),
                counters=policy._live_totals(),logits_sha256=tensor_hash(out.logits)))
        return dict(arm=arm,predicted_token_ids=predictions,frames=frames,selections=selections,
            selected_layers=getattr(selector,'selected_layers',None),scanned_rows=getattr(selector,'scanned_rows',None),
            canonical_reads_during_selector=canonical_reads[0],
            host_read_bytes=sum(getattr(c,'host_read_bytes',0) for c in policy.layer_caches.values()),
            host_write_bytes=sum(getattr(c,'host_write_bytes',0) for c in policy.layer_caches.values()),
            host_initial_write_bytes=sum(getattr(c,'host_initial_write_bytes',0) for c in policy.layer_caches.values()))
    finally:
        CpuSlotCache._canonical=original_canonical;selector.close();capture.close()
        for handle in observers:handle.remove()
        policy.reset();past=out=policy=None;gc.collect();torch.cuda.synchronize()
        # Transformers installs its own attention capture hooks lazily, outside root/selector ownership.
        root_before=before[0];assert (len(model._forward_pre_hooks),len(model._forward_hooks))==root_before
        for layer in model.model.layers:assert not layer.self_attn._forward_pre_hooks


@torch.inference_mode()
def main():
    base=Path(__file__).resolve().parent;output=base/'gpu_v2.json';assert not output.exists()
    report=dict(passed=False,runtime=runtime_identity(),arms=[])
    try:
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        prompt=Path('results/remediation_v3/latency_prompt.txt').read_text()
        teachers=json.loads(Path('results/shadow_ablation_full/fixed_stream_20260929.json').read_text())['config']['teacher_token_ids'][:12]
        ids=tokenizer(prompt,truncation=False,return_tensors='pt').input_ids[:,:4096].to('cuda')
        report.update(context=4096,decode=12,input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),teacher_token_ids=teachers)
        for arm in ('gpu_original','gpu_paired','cpu_paired'):
            report['arms'].append(run(model,ids,teachers,arm));atomic_json(output,report)
            print(arm,'finished',flush=True)
        control=report['arms'][0]
        for arm in report['arms'][1:]:
            for field in ('predicted_token_ids','frames','selections'):
                assert arm[field]==control[field],(arm['arm'],field)
            assert arm['canonical_reads_during_selector']==0
            assert arm['selected_layers']>0
        report.update(passed=True,hooks_canonical_method_restored=True,
            limits='Untimed4K12 single model/seed exact-state/choice pilot; no16Kresource/quality/timing/ordinaryCLI integration proof.')
    except BaseException as e:report['error']=f'{type(e).__name__}: {e}';raise
    finally:atomic_json(output,report)
    print('All three arms exact12 predictions/13states/384visibleKV/choices; native no canonical reads',flush=True)


if __name__=='__main__':main()
