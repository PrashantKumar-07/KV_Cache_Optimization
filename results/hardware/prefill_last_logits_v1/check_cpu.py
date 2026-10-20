"""Tiny real Mistral: caller branches, cache parity and output shapes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from candidate import compile_caller

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
import longbench_eval as bench
spec = importlib.util.spec_from_file_location('lifetime_fixture', ROOT / 'tests/test_prefill_output_lifetime.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def main():
    output = BASE / 'cpu_checks.json'
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for chunk, maximum, eos in ((0, 3, False), (4, 3, False), (0, 1, False),
            (4, 1, False), (0, 3, True), (4, 3, True), (0, 0, False)):
        pair = {}
        for last in (False, True):
            torch.manual_seed(0)
            model = fixture.MistralForCausalLM(fixture.MistralConfig(hidden_size=16,
                intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, vocab_size=32, sliding_window=None,
                attn_implementation='eager')).eval()
            tokenizer = fixture.Tokenizer()
            policy = bench.TieredKVPolicy(sram_budget=16, stt_budget=16,
                start_size=1, recent_size=2, prefill_chunk_size=chunk)
            shapes = []; predicted = []

            def after(module, positional, options, out):
                shapes.append(list(out.logits.shape))
                predicted.append(out.logits[:, -1:].argmax(-1).item())
                if eos and not options.get('output_attentions'):
                    tokenizer.eos_token_id = predicted[-1]
                expected = 1 if last or options.get('output_attentions') else positional[0].shape[1]
                assert out.logits.shape == (1, expected, 32)

            handle = model.register_forward_hook(after, with_kwargs=True)
            try:
                generate = compile_caller('longbench_eval.py', 'generate_with_policy', bench.__dict__, last)
                text = generate(model, tokenizer, 'prompt', maximum, policy, 'cpu')
            finally:
                handle.remove()
            state = hashlib.sha256()
            for key, value in policy.resident_tuple() if maximum else ():
                state.update(key.contiguous().view(torch.uint8).numpy().tobytes())
                state.update(value.contiguous().view(torch.uint8).numpy().tobytes())
            pair[last] = dict(prediction=text, forward_predictions=predicted,
                generation=getattr(policy, 'last_generation', {}), counters=policy._live_totals(),
                state_sha256=state.hexdigest(), shapes=shapes)
        for field in ('prediction', 'forward_predictions', 'generation', 'counters', 'state_sha256'):
            assert pair[False][field] == pair[True][field], (chunk, maximum, eos, field)
        rows.append(dict(chunk=chunk, maximum=maximum, eos=eos, control=pair[False], candidate=pair[True]))
    result = dict(passed=True, cases=len(rows), rows=rows,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('7 real-Mistral caller cases preserve predictions/cache/counters; shapes verified.')


if __name__ == '__main__':
    main()
