"""Tiny real-Mistral full-attention equivalence and exceptional ownership guards."""
import gc
import json
from pathlib import Path
import sys
import torch
from transformers import MistralConfig, MistralForCausalLM
from transformers.utils.output_capturing import maybe_install_capturing_hooks
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments'))
from candidate import EarlyGpuPolicy, PrefillLease, EarlyCache, PLACEMENT_AST_SHA256
from compare_cpu_batched import digest
from evaluate_query_selection import PromptQueryCapture
from longbench_eval import TieredKVPolicy, _attn_impl, to_tuple_kv, apply_tuple_kv_inplace


def hooks(model):
    return [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in model.modules()]


@torch.inference_mode()
def run(model, ids, early, hot, cold, last):
    policy = (EarlyGpuPolicy if early else TieredKVPolicy)(sram_budget=hot,
        stt_budget=cold, start_size=1, recent_size=4, kv_dtype=torch.float32,
        sttram_bifurcation_frac=.75)
    before = hooks(model)
    capture = PromptQueryCapture(model, policy)
    lease = PrefillLease(model, policy) if early else None
    records = []; full = []
    def attention_done(module, args, options, out):
        full.append(tuple(options['hidden_states'].shape))
    observers = [m.self_attn.register_forward_hook(attention_done, with_kwargs=True)
                 for m in model.model.layers]
    try:
        with _attn_impl(model, 'sdpa'):
            out = model(ids, use_cache=True, position_ids=torch.arange(ids.shape[1])[None],
                        **({'logits_to_keep': 1} if last else {}))
        logits = out.logits.clone()
        past = out.past_key_values
        assert len(full) == len(model.model.layers)
        if early:
            assert isinstance(past, EarlyCache)
            assert all(row['raw_shape'][2] == ids.shape[1] for row in policy.placement_rows)
            assert policy.placed_layers == list(range(len(model.model.layers)))
            assert not policy.prompt_q_by_layer
        apply_tuple_kv_inplace(past, policy(to_tuple_kv(past)))
        records.append((digest(policy.layer_caches), policy._live_totals(),
                        tuple((k.clone(), v.clone()) for k,v in to_tuple_kv(past))))
        capture.close()
        for step in range(5):
            out = model(torch.tensor([[step+1]]), past_key_values=past, use_cache=True,
                position_ids=torch.tensor([[ids.shape[1]+step]]), output_attentions=True)
            past=out.past_key_values
            policy.update_scores(to_tuple_kv(past),out.attentions)
            apply_tuple_kv_inplace(past,policy(to_tuple_kv(past)))
            records.append((digest(policy.layer_caches), policy._live_totals(),
                            tuple((k.clone(),v.clone()) for k,v in to_tuple_kv(past))))
            records[-1] += (out.logits.clone(),)
        return logits,records
    finally:
        capture.close()
        for observer in observers:observer.remove()
        if lease:lease.close()
        policy.reset()
        assert hooks(model)==before


def main():
    path=Path(__file__).with_name('cpu.json')
    assert not path.exists()
    torch.set_num_threads(2);torch.manual_seed(0)
    model=MistralForCausalLM(MistralConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,
        vocab_size=64,sliding_window=None,attn_implementation='eager')).eval()
    maybe_install_capturing_hooks(model.model)
    rows=[]
    for seq,hot,cold in ((8,8,8),(32,8,16),(32,12,24),(64,12,24)):
        ids=(torch.arange(seq)%60+1)[None]
        for last in (False,True):
            a=run(model,ids,False,hot,cold,last);b=run(model,ids,True,hot,cold,last)
            assert torch.equal(a[0],b[0]),'Prefill logits changed'
            for x,y in zip(a[1],b[1]):
                assert x[:2]==y[:2],'Cache/state/counters changed'
                assert all(torch.equal(i,j) for xp,yp in zip(x[2],y[2]) for i,j in zip(xp,yp))
                if len(x)==4:assert torch.equal(x[3],y[3]),'Decode logits changed'
            rows.append(dict(seq=seq,hot=hot,cold=cold,last_logits=last,exact=True))
    # Throw after layer0 has already placed KV; the root always-call hook must reset.
    baseline=hooks(model);policy=EarlyGpuPolicy(sram_budget=8,stt_budget=16,start_size=1,recent_size=4)
    capture=PromptQueryCapture(model,policy);lease=PrefillLease(model,policy)
    def fail(*args):raise RuntimeError('deliberate partial prefill')
    handle=model.model.layers[1].register_forward_pre_hook(fail)
    try:
        try:
            model(torch.arange(1,17)[None],use_cache=True)
        except RuntimeError as error:
            assert str(error)=='deliberate partial prefill'
        else:raise AssertionError('Expected injected exception')
        assert not policy.layer_caches and not policy.prompt_q_by_layer and not policy.placed_layers
    finally:
        handle.remove();lease.close();capture.close();policy.reset()
    assert hooks(model)==baseline
    # Out-of-order updates fail before state changes.
    policy=EarlyGpuPolicy(sram_budget=8,stt_budget=16,start_size=1,recent_size=4)
    policy.prefill_layers=3;policy.prefill_sequence=16
    try:policy.place(1,torch.zeros(1,2,16,8),torch.zeros(1,2,16,8))
    except ValueError:pass
    else:raise AssertionError('Expected ordered-layer validation')
    assert not policy.layer_caches and not policy.placed_layers
    gc.collect()
    report=dict(passed=True,pairs=rows,partial_exception_reset=True,order_guard=True,
                all_diagnostic_hooks_removed=True,placement_ast_sha256=PLACEMENT_AST_SHA256,
                limits='CPU core-cache tiny-model validation; direct pinned/native GPU backend remains untested.')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print('Eight CPU pairs exact; partial-error reset/order/hook cleanup pass',flush=True)


if __name__=='__main__':main()
