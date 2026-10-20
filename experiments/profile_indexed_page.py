"""One descriptive cProfile trace of the indexed full model+policy loop."""
import cProfile
import hashlib
import io
from pathlib import Path
import pstats

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_query_selection import PromptQueryCapture, QuerySelector
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import atomic_json, runtime_identity
from stable_page_index import IndexedTieredKVCache


def main():
    model_path = '../models/mistral-7b-instruct'
    prompt_path = Path('results/remediation_v3/latency_prompt.txt')
    prompt = prompt_path.read_text()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy = TieredKVPolicy(sram_budget=992, stt_budget=3104, stt_expose_quota=32,
        start_size=4, recent_size=128, kv_dtype=torch.bfloat16,
        sttram_bifurcation_frac=.75)
    policy.cache_cls = IndexedTieredKVCache
    selector = QuerySelector(model, policy, candidate_pages=32, use_index=True)
    capture = PromptQueryCapture(model, policy)
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        timing = profile_one(model, tokenizer, policy, 8192, n_decode=5,
                             n_warmup=0, n_repeat=1, prompt_text=prompt)
        profiler.disable()
    finally:
        selector.close()
        capture.close()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(80)
    output = Path('results/page_recall/indexed_cprofile_8k.txt')
    output.write_text(stream.getvalue())
    atomic_json('results/page_recall/indexed_cprofile_8k.json', {
        'config': {'model': model_path, 'context': 8192, 'decode': 5,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runtime': runtime_identity(), 'scope': 'one descriptive cProfile; GPU waits included'},
        'timing': timing, 'profile_text_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
    print(timing['itl_ms'], 'ms/token', output)


if __name__ == '__main__':
    main()
