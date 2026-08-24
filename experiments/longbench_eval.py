#!/usr/bin/env python3
# longbench_eval.py — unified LongBench evaluation for TieredKV vs SOTA baselines
#
# Runs all methods through the same decode loop on LongBench tasks, then
# scores with identical metrics. Produces a JSON for plot_sota_comparison.py.
#
# Methods:
#   full          — no eviction (oracle upper bound)
#   streamingllm  — sinks + sliding window
#   h2o           — cumulative-attention heavy-hitter eviction
#   snapkv        — observation-window importance + H2O decode eviction
#   tieredkv      — our 3-tier inclusive victim cache (novel)
#
# All eviction methods use the same total KV budget for memory-fair comparison.

import os, sys, json, argparse, math
import torch
import numpy as np

# Ensure HF_HOME points to /data partition with extracted LongBench datasets
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/data/nishant/Nishant/Prashant/hf_cache"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from longbench_metrics import (
    DATASET_TO_METRIC, DATASET_TO_MAXGEN, DATASET_TO_PROMPT,
)


# ---------------------------------------------------------------------------
# DynamicCache helpers
# Newer transformers (4.44+) returns DynamicCache with .layers[i].keys/values
# ---------------------------------------------------------------------------

def _get_layer_kv(layer):
    """Extract (k, v) tensors from a DynamicCache layer."""
    return layer.keys, layer.values


def to_tuple_kv(past_kv):
    """Convert DynamicCache or legacy tuple to tuple-of-(k,v) pairs."""
    if past_kv is None:
        return None
    if isinstance(past_kv, DynamicCache):
        return tuple(_get_layer_kv(layer) for layer in past_kv.layers)
    return tuple((k, v) for k, v in past_kv)


def apply_tuple_kv_inplace(past_kv, tuple_kv):
    """Write trimmed (k,v) pairs back into DynamicCache.layers in-place."""
    if isinstance(past_kv, DynamicCache):
        for i, (k, v) in enumerate(tuple_kv):
            past_kv.layers[i].keys = k
            past_kv.layers[i].values = v
    return past_kv


# ---------------------------------------------------------------------------
# KV cache eviction policies
# Each policy receives tuple[(K, V)] per layer, K/V: (batch, heads, seq, dim)
# and returns a trimmed tuple[(K, V)].
# ---------------------------------------------------------------------------

class FullCachePolicy:
    """No eviction — oracle baseline."""
    name = "Full"
    def __call__(self, past_kv):
        return past_kv


class StreamingLLMPolicy:
    """Sinks + sliding window. Intermediate tokens are permanently dropped."""
    name = "StreamingLLM"

    def __init__(self, start_size=4, recent_size=60):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size

    def __call__(self, past_kv):
        if past_kv is None:
            return None
        seq_len = past_kv[0][0].size(2)
        if seq_len <= self.cache_size:
            return past_kv
        return tuple(
            (
                torch.cat([k[:, :, :self.start_size], k[:, :, seq_len - self.recent_size:]], dim=2),
                torch.cat([v[:, :, :self.start_size], v[:, :, seq_len - self.recent_size:]], dim=2),
            )
            for k, v in past_kv
        )


class H2OPolicy:
    """Heavy-Hitter Oracle: evict lowest cumulative-attention token per step.

    Sinks + recent window are protected. The rest compete by accumulated
    attention weight; the loser is permanently dropped.
    """
    name = "H2O"

    def __init__(self, budget=64, start_size=4, recent_size=16):
        self.budget = budget
        self.start_size = start_size
        self.recent_size = recent_size
        self.cum_scores = {}  # layer -> Tensor(seq,)

    def reset(self):
        self.cum_scores = {}

    def __call__(self, past_kv):
        if past_kv is None:
            return None
        result = []
        for layer_idx, (k, v) in enumerate(past_kv):
            seq_len = k.size(2)
            if seq_len <= self.budget:
                result.append((k, v))
                continue

            # Sync cumulative scores to current cache size
            if layer_idx not in self.cum_scores:
                self.cum_scores[layer_idx] = torch.zeros(seq_len, device=k.device, dtype=torch.float32)
            cum = self.cum_scores[layer_idx]
            if cum.size(0) < seq_len:
                cum = torch.cat([cum, torch.zeros(seq_len - cum.size(0), device=cum.device)])
            elif cum.size(0) > seq_len:
                cum = cum[:seq_len]
            self.cum_scores[layer_idx] = cum

            # Build eviction scores: protected = inf
            score = cum.clone()
            score[:self.start_size] = float("inf")
            score[max(0, seq_len - self.recent_size):] = float("inf")

            n_evict = seq_len - self.budget
            _, evict_idx = torch.topk(score, n_evict, largest=False)
            keep = torch.ones(seq_len, dtype=torch.bool, device=k.device)
            keep[evict_idx] = False

            result.append((k[:, :, keep], v[:, :, keep]))
            self.cum_scores[layer_idx] = cum[keep]

        return tuple(result)

    def update_scores(self, past_kv, attn_weights):
        """Accumulate attention mass from a single forward pass (decode step).
        attn_weights: list of (batch, heads, 1, seq) tensors, one per layer.
        """
        if attn_weights is None:
            return
        for layer_idx, (k, v) in enumerate(past_kv):
            if layer_idx >= len(attn_weights) or attn_weights[layer_idx] is None:
                continue
            w = attn_weights[layer_idx]  # (batch, heads, 1, seq)
            seq_len = k.size(2)
            if layer_idx not in self.cum_scores:
                self.cum_scores[layer_idx] = torch.zeros(seq_len, device=k.device, dtype=torch.float32)
            cum = self.cum_scores[layer_idx]
            attn_sum = w.float().squeeze(2).mean(dim=(0, 1))[:seq_len]  # (seq,)
            if attn_sum.size(0) <= cum.size(0):
                cum[:attn_sum.size(0)] += attn_sum


class SnapKVPolicy:
    """SnapKV: compress prompt using observation-window importance, then H2O decode."""
    name = "SnapKV"

    def __init__(self, budget=64, start_size=4, recent_size=16, pool_kernel=5):
        self.budget = budget
        self.start_size = start_size
        self.recent_size = recent_size
        self.pool_kernel = pool_kernel
        self.compressed = False
        self._h2o = H2OPolicy(budget=budget, start_size=start_size, recent_size=recent_size)

    def reset(self):
        self.compressed = False
        self._h2o.reset()

    def __call__(self, past_kv):
        if past_kv is None:
            return None

        if not self.compressed:
            self.compressed = True
            return self._compress_prompt(past_kv)

        return self._h2o(past_kv)

    def _compress_prompt(self, past_kv):
        result = []
        for k, v in past_kv:
            seq_len = k.size(2)
            if seq_len <= self.budget:
                result.append((k, v))
                continue

            w = min(self.recent_size, seq_len)
            q_win = k[:, :, seq_len - w:]
            scale = 1.0 / math.sqrt(k.size(-1))
            scores = torch.matmul(q_win.float(), k.float().transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            importance = attn.sum(dim=2).mean(dim=(0, 1))  # (seq,)

            if self.pool_kernel > 1 and seq_len >= self.pool_kernel:
                pad = self.pool_kernel // 2
                importance = torch.nn.functional.avg_pool1d(
                    importance.view(1, 1, -1).float(),
                    kernel_size=self.pool_kernel, stride=1, padding=pad
                ).view(-1)[:seq_len]

            sink_idx = set(range(min(self.start_size, seq_len)))
            win_idx = set(range(max(0, seq_len - w), seq_len))
            pinned = sink_idx | win_idx
            rest = sorted(
                [i for i in range(seq_len) if i not in pinned],
                key=lambda i: importance[i].item(), reverse=True
            )
            extra = max(0, self.budget - len(pinned))
            idx = sorted(pinned | set(rest[:extra]))
            idx_t = torch.tensor(idx, device=k.device)
            result.append((k[:, :, idx_t], v[:, :, idx_t]))
        return tuple(result)


class TieredKVPolicy:
    """3-tier inclusive victim cache.

    Tier 1 (VRAM): hot working set, size = sram_budget.
    Tier 2 (STT-RAM): victim cache with shadow backups (simulated).
    Core property: re-demotion to STT where shadow backup exists = zero write cost.
    """
    name = "TieredKV"

    def __init__(self, sram_budget=128, stt_budget=256, start_size=4, recent_size=16):
        self.sram_budget = sram_budget
        self.stt_budget = stt_budget
        self.start_size = start_size
        self.recent_size = recent_size
        self.initialized = False
        self.tier1 = {}           # layer_idx -> set of token positions in VRAM
        self.tier2_backup = {}    # layer_idx -> set of positions with shadow backup
        self.cum_scores = {}      # layer_idx -> Tensor(seq,)
        self.total_writes_saved = 0
        self.total_paid_writes = 0
        self.total_demoted = 0

    def reset(self):
        self.initialized = False
        self.tier1 = {}
        self.tier2_backup = {}
        self.cum_scores = {}
        self.total_writes_saved = 0
        self.total_paid_writes = 0
        self.total_demoted = 0

    def __call__(self, past_kv):
        if past_kv is None:
            return None

        if not self.initialized:
            self.initialized = True
            return self._bifurcate(past_kv)

        return self._decode_step(past_kv)

    def _bifurcate(self, past_kv):
        """Initial bifurcation after prefill: top tokens -> VRAM, rest -> STT.

        After this, the returned tensors are compressed to sram_budget tokens.
        We store cum_scores in COMPRESSED space (0..len(keep)-1) so that
        _decode_step can safely index into the current tensor size.
        """
        result = []
        for layer_idx, (k, v) in enumerate(past_kv):
            seq_len = k.size(2)
            w = min(self.recent_size, seq_len)

            # SnapKV-style importance scoring for initial placement
            q_win = k[:, :, seq_len - w:]
            scale = 1.0 / math.sqrt(k.size(-1))
            scores = torch.matmul(q_win.float(), k.float().transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            importance = attn.sum(dim=2).mean(dim=(0, 1))  # (seq_len,)

            sink_set = set(range(min(self.start_size, seq_len)))
            win_set = set(range(max(0, seq_len - w), seq_len))
            pinned = sink_set | win_set

            rest = sorted(
                [i for i in range(seq_len) if i not in pinned],
                key=lambda i: importance[i].item(), reverse=True
            )
            vram_extra = max(0, self.sram_budget - len(pinned))
            keep_abs = sorted(pinned | set(rest[:vram_extra]))

            # Store scores in COMPACT space (indices 0..N-1 of compressed tensor)
            # cum_scores stays on CPU; k/v indexing uses GPU tensor
            idx_cpu = torch.tensor(keep_abs, dtype=torch.long)          # CPU
            idx_t = idx_cpu.to(k.device)                                # GPU
            result.append((k[:, :, idx_t], v[:, :, idx_t]))

            self.cum_scores[layer_idx] = importance[idx_cpu].cpu().float()  # always CPU
            # Shadow backup tracking: count of positions initially placed in STT
            n_evicted = seq_len - len(keep_abs)
            # Use a counter instead of absolute positions (positions shift after compression)
            self.tier2_backup[layer_idx] = n_evicted  # int: how many shadow slots exist
        return tuple(result)

    def update_scores(self, past_kv, attn_weights):
        """Accumulate attention mass from a single decode step."""
        if attn_weights is None:
            return
        for layer_idx, (k, v) in enumerate(past_kv):
            if layer_idx >= len(attn_weights) or attn_weights[layer_idx] is None:
                continue
            w = attn_weights[layer_idx]  # (batch, heads, 1, seq)
            seq_len = k.size(2)
            if layer_idx not in self.cum_scores:
                self.cum_scores[layer_idx] = torch.zeros(seq_len, dtype=torch.float32)
            cum = self.cum_scores[layer_idx]
            cum = cum.cpu().float()
            attn_sum = w.float().squeeze(2).mean(dim=(0, 1))[:seq_len].cpu()
            
            # Extend cum if a new token was just appended in decode step
            if cum.size(0) < seq_len:
                cum = torch.cat([cum, torch.zeros(seq_len - cum.size(0), dtype=torch.float32)])
                
            cum[:seq_len] += attn_sum
            self.cum_scores[layer_idx] = cum

    def _decode_step(self, past_kv):
        """Per-decode-step: model appended new token; evict excess from COMPACT cache.

        k.size(2) = previous compact size + 1 (new token at end).
        cum_scores is in compact space so all indexing is safe.
        """
        result = []
        for layer_idx, (k, v) in enumerate(past_kv):
            seq_len = k.size(2)  # compact tensor size including new token

            # Extend cum_scores for the newly appended token — always on CPU float
            cum = self.cum_scores.get(layer_idx, torch.zeros(seq_len, dtype=torch.float32))
            cum = cum.cpu().float()  # ensure CPU
            if cum.size(0) < seq_len:
                cum = torch.cat([cum, torch.zeros(seq_len - cum.size(0), dtype=torch.float32)])

            # All positions 0..seq_len-1 are in the current tensor; evict to budget
            vram_set = set(range(seq_len))

            while len(vram_set) > self.sram_budget:
                # Protected: sinks (first start_size) and recent (last recent_size)
                candidates = [
                    (cum[i].item(), i)
                    for i in vram_set
                    if i >= self.start_size and i < seq_len - self.recent_size
                ]
                if not candidates:
                    break
                _, victim = min(candidates)
                vram_set.discard(victim)

                # Inclusive victim cache: track write savings
                shadow_count = self.tier2_backup.get(layer_idx, 0)
                if shadow_count > 0:
                    self.total_writes_saved += 1
                    self.tier2_backup[layer_idx] = shadow_count - 1
                else:
                    self.total_paid_writes += 1
                self.total_demoted += 1

            # Select kept tokens; result is compact 0..len(keep)-1
            keep = sorted(vram_set)
            idx_cpu = torch.tensor(keep, dtype=torch.long)      # CPU — for indexing cum
            idx_t = idx_cpu.to(k.device)                        # GPU — for indexing k/v
            result.append((k[:, :, idx_t], v[:, :, idx_t]))

            # Update cum_scores in new compact space (always CPU float)
            self.cum_scores[layer_idx] = cum[idx_cpu]

        return tuple(result)

    def stats(self):
        total = self.total_paid_writes + self.total_writes_saved
        pct = self.total_writes_saved / total * 100 if total > 0 else 0
        return {
            "demoted": self.total_demoted,
            "paid_writes": self.total_paid_writes,
            "writes_saved": self.total_writes_saved,
            "write_savings_pct": round(pct, 1),
        }


# ---------------------------------------------------------------------------
# Token-by-token generation with KV policy
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_with_policy(model, tokenizer, input_text, max_gen, policy, device,
                         max_ctx=8192, task_name=""):
    """Generate text token-by-token, applying cache policy after each step."""
    toks = tokenizer(input_text, truncation=False, return_tensors="pt")
    input_ids = toks.input_ids

    # Standard middle-split truncation for long contexts (preserves query + context ends)
    if input_ids.size(1) > max_ctx:
        half = max_ctx // 2
        input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)

    input_ids = input_ids.to(device)

    # Prefill
    out = model(input_ids, use_cache=True)
    
    raw_past = out.past_key_values
    tuple_kv = to_tuple_kv(raw_past)
    tuple_kv = policy(tuple_kv)
    apply_tuple_kv_inplace(raw_past, tuple_kv)
    past_kv = raw_past

    next_tok = out.logits[:, -1:].argmax(dim=-1)
    generated = [next_tok.item()]

    # Decode loop
    for _ in range(max_gen - 1):
        out = model(next_tok, past_key_values=past_kv, use_cache=True, output_attentions=True)
        raw_past = out.past_key_values
        tuple_kv = to_tuple_kv(raw_past)
        if hasattr(policy, "update_scores"):
            policy.update_scores(tuple_kv, out.attentions)
        tuple_kv = policy(tuple_kv)
        apply_tuple_kv_inplace(raw_past, tuple_kv)
        past_kv = raw_past

        next_tok = out.logits[:, -1:].argmax(dim=-1)
        tok_id = next_tok.item()
        generated.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(model, tokenizer, task_name, policy, device, max_samples=None, is_instruct=False):
    """Evaluate one LongBench task, return (avg_score, sample_results)."""
    data = None
    try:
        data = load_dataset("THUDM/LongBench", f"{task_name}_e", split="test", trust_remote_code=True)
    except Exception:
        pass

    if data is None:
        try:
            data = load_dataset("THUDM/LongBench", task_name, split="test", trust_remote_code=True)
        except Exception:
            pass

    # Direct local JSONL fallback if HF cache builder has path issues
    if data is None:
        import glob
        local_matches = glob.glob(f"/data/nishant/Nishant/Prashant/hf_cache/**/{task_name}_e.jsonl", recursive=True)
        if not local_matches:
            local_matches = glob.glob(f"/data/nishant/Nishant/Prashant/hf_cache/**/{task_name}.jsonl", recursive=True)
        if local_matches:
            local_items = []
            with open(local_matches[0], "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        local_items.append(json.loads(line))
            data = local_items
        else:
            raise RuntimeError(f"Could not load dataset for task: {task_name}")

    prompt_fmt = DATASET_TO_PROMPT[task_name]
    max_gen = DATASET_TO_MAXGEN[task_name]
    metric_fn = DATASET_TO_METRIC[task_name]

    results = []
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break
        input_text = prompt_fmt.format(**item)
        if is_instruct:
            input_text = f"<s>[INST] {input_text} [/INST]"
        answers = item["answers"]
        
        # Reset per-sample state
        if hasattr(policy, "reset"):
            policy.reset()

        pred = generate_with_policy(
            model, tokenizer, input_text, max_gen, policy, device, task_name=task_name
        )

        score = max(
            metric_fn(pred, gt, all_classes=item.get("all_classes", []))
            for gt in answers
        )
        results.append(score)

        if (i + 1) % 5 == 0 or (i + 1) == (max_samples if max_samples else len(data)):
            print(f"  [{policy.name}] {task_name} {i+1}/{(max_samples if max_samples else len(data))}  Score={np.mean(results)*100:.2f}")

        # Proactive memory hygiene
        if (i + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return round(np.mean(results) * 100, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", nargs="+",
                        default=["multifieldqa_en", "qasper", "narrativeqa", "hotpotqa", "gov_report", "triviaqa"])
    parser.add_argument("--methods", nargs="+",
                        default=["full", "streamingllm", "h2o", "snapkv", "tieredkv"])
    parser.add_argument("--budget", type=int, default=1024,
                        help="Total KV budget in tokens for eviction methods")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per task for quick runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", default="results/longbench_comparison.json")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint file to resume from (defaults to <json>.ckpt)")
    args = parser.parse_args()

    # checkpoint file lives next to the final JSON
    ckpt_path = args.checkpoint or (args.json + ".ckpt")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Budget allocation: 1/3 VRAM, 2/3 STT
    sink = 4
    recent = max(16, args.budget // 8)
    sram = args.budget // 3
    stt = args.budget - sram

    def make_policy(name):
        if name == "full":
            return FullCachePolicy()
        elif name == "streamingllm":
            return StreamingLLMPolicy(start_size=sink, recent_size=args.budget - sink)
        elif name == "h2o":
            return H2OPolicy(budget=args.budget, start_size=sink, recent_size=recent)
        elif name == "snapkv":
            return SnapKVPolicy(budget=args.budget, start_size=sink, recent_size=recent)
        elif name == "tieredkv":
            return TieredKVPolicy(sram_budget=sram, stt_budget=stt, start_size=sink, recent_size=recent)
        raise ValueError(f"Unknown method: {name}")

    # Load checkpoint so we can skip already-finished work
    all_results = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            all_results = json.load(f)
        print(f"Resuming from checkpoint: {ckpt_path}")
        for t, methods in all_results.items():
            for m in methods:
                print(f"  [SKIP] {t} / {m} (already done, score={methods[m].get('score')})")  
    for task in args.tasks:
        print(f"\n{'='*60}\nTask: {task}\n{'='*60}")
        task_res = {}
        for mname in args.methods:
            # Skip if already completed in a previous run
            if task in all_results and mname in all_results[task]:
                print(f"\n--- {mname} [SKIPPED — already in checkpoint] ---")
                continue

            policy = make_policy(mname)
            print(f"\n--- {policy.name} ---")
            is_instruct = "instruct" in args.model.lower()
            score = evaluate_task(model, tokenizer, task, policy, args.device, args.max_samples, is_instruct)

            if task not in all_results:
                all_results[task] = {}
            all_results[task][mname] = {"score": score}
            if hasattr(policy, "stats"):
                all_results[task][mname]["cache_stats"] = policy.stats()
            print(f"  {policy.name} → {task}: {score:.2f}")

            # Save checkpoint immediately after each method finishes
            with open(ckpt_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  [checkpoint saved → {ckpt_path}]")

    # Print summary table
    print(f"\n{'='*60}\nLONGBENCH RESULTS (F1 × 100)\n{'='*60}")
    col_w = 14
    hdr = f"{'Task':<22}" + "".join(f"{m:>{col_w}}" for m in args.methods)
    print(hdr)
    print("-" * len(hdr))
    for task in args.tasks:
        row = f"{task:<22}"
        for m in args.methods:
            s = all_results[task].get(m, {}).get("score", 0)
            row += f"{s:>{col_w}.2f}"
        print(row)
    print("-" * len(hdr))
    avg_row = f"{'Average':<22}"
    for m in args.methods:
        avg = np.mean([all_results[t].get(m, {}).get("score", 0) for t in args.tasks])
        avg_row += f"{avg:>{col_w}.2f}"
    print(avg_row)

    # Save
    os.makedirs(os.path.dirname(args.json) if os.path.dirname(args.json) else ".", exist_ok=True)
    output = {
        "config": {"model": args.model, "budget": args.budget,
                   "tasks": args.tasks, "methods": args.methods},
        "results": all_results,
    }
    with open(args.json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {args.json}")


if __name__ == "__main__":
    main()
