"""Source-grounded payload accounting for the official Mistral FlexiCache setup."""
import ast
import argparse
import hashlib
import json
import resource
import subprocess
from pathlib import Path

from provenance import atomic_json


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/data/nishant/tieredkv_external/flexicache")
MODEL = ROOT.parent / "models/mistral-7b-instruct/config.json"
OUT = ROOT / "results/baseline_fidelity/flexicache_resource_preflight_v2_20261001.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve prior evidence: {args.output}")
    if subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Expected an unchanged released checkout")
    script = UPSTREAM / "benchmarks/FlexiCache/Language_Modelling/LongBench/Mistral7b.sh"
    source = script.read_text()
    for expected in ("--num_unstable_heads 64", "--rerank_frequency 16",
                     "--topK_budget 64"):
        if expected not in source:
            raise RuntimeError(f"Official Mistral setting changed: {expected}")
    data_path = UPSTREAM / "vllm/v1/flexicache/model_data.py"
    data = ast.literal_eval(ast.parse(data_path.read_text()).body[0].value)
    profile = data["mistralai/Mistral-7B-Instruct-v0.2"][
        "unstable-64-profile-gov_report-topk-64"]
    heads = {tuple(pair) for pair in profile}
    model = json.loads(MODEL.read_text())
    layers = model["num_hidden_layers"]
    kv_heads = model["num_key_value_heads"]
    head_dim = model["hidden_size"] // model["num_attention_heads"]
    if len(profile) != 64 or len(heads) != 64:
        raise RuntimeError("Unstable-head profile count or uniqueness changed")
    if any(not 0 <= layer < layers or not 0 <= head < kv_heads for layer, head in heads):
        raise RuntimeError("Profile contains out-of-range head")
    stable = layers * kv_heads - len(heads)
    bytes_per_head_token = 2 * head_dim * 2  # K + V, bf16
    prompt_tokens, page_tokens, top_pages = 15789, 16, 64
    completed = prompt_tokens // page_tokens * page_tokens
    host_payload = stable * completed * bytes_per_head_token
    gpu_unstable = len(heads) * prompt_tokens * bytes_per_head_token
    gpu_stable = stable * min(prompt_tokens, top_pages * page_tokens) * bytes_per_head_token
    tiered_fast = layers * kv_heads * 1024 * bytes_per_head_token
    tiered_slow_capacity = layers * kv_heads * 2048 * bytes_per_head_token
    tiered_initial_slow = layers * kv_heads * 1536 * bytes_per_head_token
    cfg_path = UPSTREAM / "vllm/v1/flexicache/config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg["cpu_kv_cache_size"] != 180:
        raise RuntimeError("Official CPU pool changed")
    d2h = (UPSTREAM / "vllm/v1/worker/gpu_model_runner.py").read_text()
    if not all(k in d2h for k in ("last_tx_blk_np", "total_full_blk_np",
                                    "offload_kv_cache_d2h_selected")):
        raise RuntimeError("Offload trigger source changed")
    if 'dtype=dtype, device="cpu", pin_memory=True' not in d2h:
        raise RuntimeError("Pinned host allocation source changed")
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal:", "MemAvailable:"))}
    source_paths = [
        "README.md", "pyproject.toml", "requirements/cuda.txt", "CMakeLists.txt",
        "vllm/v1/flexicache/config.py", "vllm/v1/core/kv_cache_utils.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/flexicache/kernels/cuda/kv_d2h_tx_gpu_mapped.cu",
        "vllm/v1/flexicache/kernels/cuda/h2d_copy_blocks.cu"]
    report = {
        "scope": "analytical KV payload from released settings; not measured allocation or traffic",
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "official_script_sha256": sha(script), "profile_sha256": sha(data_path),
        "config_sha256": sha(cfg_path), "model_config_sha256": sha(MODEL),
        "audited_source_sha256": {path: sha(UPSTREAM / path) for path in source_paths},
        "host_memory_snapshot_bytes": memory,
        "host_memlock_limit_bytes": list(resource.getrlimit(resource.RLIMIT_MEMLOCK)),
        "gpu_snapshot": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
            "--format=csv,noheader"], text=True),
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "official_pinned_pool_fits_available_host_memory": cfg["cpu_kv_cache_size"] * 1024**3 < memory["MemAvailable"],
        "full_pinned_pool_allocation_tested": False,
        "local_feasibility_decision": (
            "Host RAM capacity screen passes; full pinned-pool allocation, modified-vLLM "
            "build and model execution remain unverified. Use an isolated torch 2.6.0 "
            "environment and CUDA architecture 8.9 for L40S; do not replace the live "
            "TieredKV environment. GPU1 is occupied by another process."),
        "model_layers": layers, "kv_heads_per_layer": kv_heads,
        "unstable_head_count": len(heads), "stable_head_count": stable,
        "head_dim": head_dim, "bytes_per_head_token": bytes_per_head_token,
        "prompt_tokens": prompt_tokens, "page_tokens": page_tokens,
        "top_pages": top_pages, "top_tokens_per_stable_head": top_pages * page_tokens,
        "completed_prompt_tokens": completed,
        "flexicache_host_completed_kv_payload_bytes": host_payload,
        "flexicache_gpu_unstable_kv_payload_bytes": gpu_unstable,
        "flexicache_gpu_stable_topk_kv_payload_bytes": gpu_stable,
        "flexicache_gpu_kv_payload_sum_bytes": gpu_unstable + gpu_stable,
        "flexicache_gpu_plus_host_payload_sum_bytes": host_payload + gpu_unstable + gpu_stable,
        "flexicache_declared_pinned_host_pool_bytes": cfg["cpu_kv_cache_size"] * 1024**3,
        "tieredkv_fast_capacity_payload_bytes": tiered_fast,
        "tieredkv_slow_capacity_payload_bytes": tiered_slow_capacity,
        "tieredkv_initial_slow_payload_bytes": tiered_initial_slow,
        "tieredkv_max_logical_payload_bytes": tiered_fast + tiered_slow_capacity,
        "tieredkv_initial_logical_payload_bytes": tiered_fast + tiered_initial_slow,
        "upstream_full_page_offload_evidence": (
            "gpu_model_runner.py compares completed-page counters and sends "
            "new pages to offload_kv_cache_d2h_selected; build_pairs_kernel emits "
            "only [last_tx_blk, total_full_blks) and skips CPU entries -1. "
            "Reload reads retained host pages; this is source evidence, not measured traffic."),
        "payload_limitations": (
            "Single-request analytical prompt payload only. Excludes final partial stable "
            "pages, generated tokens, page-selection details, min/max metadata, block "
            "tables, activations, temporary/prefill buffers and reserved pools. "
            "The 180 GiB pool is configured capacity, not useful occupied KV."),
        "runner_sha256": sha(Path(__file__)),
    }
    atomic_json(args.output, report)
    print(json.dumps({k: report[k] for k in (
        "flexicache_host_completed_kv_payload_bytes",
        "flexicache_gpu_kv_payload_sum_bytes",
        "flexicache_gpu_plus_host_payload_sum_bytes",
        "tieredkv_max_logical_payload_bytes")}, indent=2))


if __name__ == "__main__":
    main()
