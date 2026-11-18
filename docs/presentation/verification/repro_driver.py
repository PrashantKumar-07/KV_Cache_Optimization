"""Re-execute selected recorded requests with the frozen evaluation source and compare.

Writes only to the scratch folder; never touches the original records.
"""
import hashlib, importlib.util, json, sys
from pathlib import Path

REPRO = Path(sys.argv[1])            # extracted frozen source root
ORIG = Path(sys.argv[2])             # real kv_cache repo root
spec = importlib.util.spec_from_file_location('run_full', REPRO / 'results/full_evaluation_20261015/run_full.py')
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
bench = rf.bench
plan = json.loads((ORIG / 'results/full_evaluation_20261015/plan.json').read_text())
model_path = str(ORIG.parent / 'models/mistral-7b-instruct')
PICKS = [('multifieldqa_en', 0), ('hotpotqa', 7), ('narrativeqa', 3), ('triviaqa', 11), ('qasper', 5), ('gov_report', 2)]
ARMS = ['full', 'snapkv_adapted', 'tiered_hot992']
out = {}
torch.manual_seed(0)
with torch.inference_mode():
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 device_map='cuda', attn_implementation='eager').eval()
    probe = model(torch.tensor([[1]], device='cuda'), use_cache=False, output_attentions=True,
                  **bench.prefill_logit_kwargs(model, True)); del probe
    tok = AutoTokenizer.from_pretrained(model_path)
    for task, index in PICKS:
        rows = rf.read_jsonl(ORIG / plan['dataset_files'][task])
        item = rows[index]
        prompt = bench.DATASET_TO_PROMPT[task].format(**item)
        if task not in bench.NO_CHAT_TASKS:
            prompt = '[INST] ' + prompt + ' [/INST]'
        ids = tok(prompt, return_tensors='pt').input_ids
        if ids.shape[1] > 31500:
            ids = torch.cat([ids[:, :15750], ids[:, -15750:]], 1)
        th = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        ids = ids.to('cuda')
        for arm in ARMS:
            rec = None
            for shard in (0, 1):
                p = ORIG / f'results/full_evaluation_20261015/records/shard_{shard}/{task}_{index:04d}_{arm}.json'
                if p.exists():
                    rec = json.loads(p.read_text())
            res = rf.run_request(model, tok, ids, task, item, arm)
            same_tokens = res['generated_token_ids'] == rec['generated_token_ids']
            stats_keys = ('paid_writes', 'writes_saved', 'demoted', 'promoted')
            same_stats = all(res['cache_stats'].get(k) == rec['cache_stats'].get(k) for k in stats_keys)
            line = dict(task=task, index=index, arm=arm, input_hash_match=th == rec['input_ids_sha256'],
                        tokens_identical=same_tokens, stats_identical=same_stats,
                        score_now=round(res['score'], 4), score_recorded=round(rec['score'], 4),
                        prediction_now=res['prediction'][:80], prediction_recorded=rec['prediction'][:80])
            print(json.dumps(line), flush=True)
            out[f'{task}_{index}_{arm}'] = line
            torch.cuda.empty_cache()
(REPRO / 'repro_result.json').write_text(json.dumps(out, indent=1))
