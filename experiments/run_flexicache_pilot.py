"""One identical-input Mistral request through the released FlexiCache engine."""
import argparse
import ctypes
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from longbench_metrics import DATASET_TO_PROMPT, qa_f1_score
from provenance import atomic_json, read_jsonl, runtime_identity


MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
UPSTREAM_COMMIT = "ab0f495b10fc6509c1637af35779b29121fb7d3f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pool_info(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "pinned": tensor.is_pinned(),
            "allocated_payload_bytes": tensor.numel() * tensor.element_size()}


def pin_worker_cudart(worker, path):
    """Keep the release's priority query on the validated CUDA 12 runtime."""
    path = path.resolve(strict=True)
    library = ctypes.CDLL(str(path))
    version = ctypes.c_int()
    if library.cudaRuntimeGetVersion(ctypes.byref(version)) != 0 or version.value // 1000 != 12:
        raise ValueError("Expected a working CUDA 12 runtime")
    least, greatest = ctypes.c_int(), ctypes.c_int()
    code = library.cudaDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest))
    if code != 0 or greatest.value >= least.value:
        raise ValueError(f"Stream-priority query failed: code={code}, range={(least.value, greatest.value)}")
    original = worker.find_library
    original_lookup = original("cudart")
    worker.find_library = lambda name: str(path) if name == "cudart" else original(name)
    return {"scope": "worker cudart library resolution only; upstream source unchanged",
            "original_lookup": original_lookup, "library": str(path),
            "library_sha256": sha(path), "runtime_version": version.value,
            "query_return_code": code, "priority_range": [least.value, greatest.value]}


def observe_resources(runner, core):
    """Observe submitted copies and scheduler snapshots without GPU reads."""
    report = {"scope": "submitted payload and allocation snapshots; excludes controller traffic, energy and continuous peaks",
              "transfers": {}, "scheduler_snapshots": []}
    for direction, module, name in (
            ("d2h", runner._kv_d2h_tx, "kv_cache_d2h_transfer"),
            ("h2d", runner._kv_h2d_tx, "kv_cache_h2d_transfer")):
        counts = {"calls": 0, "paired_kv_pages": 0, "payload_bytes": 0}
        report["transfers"][direction] = counts
        original = getattr(module, name)

        def counted(*args, _original=original, _counts=counts, **kwargs):
            result = _original(*args, **kwargs)
            pages = args[2].numel()
            page_bytes = 2 * args[0].shape[2] * args[0].shape[3] * args[0].element_size()
            _counts["calls"] += 1
            _counts["paired_kv_pages"] += pages
            _counts["payload_bytes"] += pages * page_bytes
            return result

        setattr(module, name, counted)

    manager = core.scheduler.kv_cache_manager
    schedule = core.scheduler.schedule

    def observed_schedule(*args, **kwargs):
        result = schedule(*args, **kwargs)
        pools = manager.block_pools
        report["scheduler_snapshots"].append({
            "scheduled_tokens": sum(result.num_scheduled_tokens.values()),
            "gpu_pages_in_use": sum(p.num_gpu_blocks - p.get_num_free_blocks() for p in pools),
            "cpu_pages_in_use": sum(p.num_cpu_blocks - p.get_num_free_cpu_blocks() for p in pools),
            "minmax_pages_in_use": sum(p.num_blocks - p.get_num_free_blocks() for p in manager.minmax_block_pools),
            "pending_gpu_release_pages": sum(len(layer) for layers in manager.pending_free.values() for layer in layers)})
        return result

    core.scheduler.schedule = observed_schedule
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--index", type=int, default=9)
    parser.add_argument("--expected-token-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cudart-library", type=Path, help="Explicit CUDA 12 runtime for the worker's ctypes query")
    parser.add_argument("--l40s-prefill-stage-one", action="store_true", help="Explicit prefill pipeline-stage adapter verified by the L40S screen")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve prior evidence: {args.output}")
    revision = subprocess.check_output([
        "git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError("Unexpected FlexiCache release")
    report = {"scope": "one identical-input feasibility request; not resource-matched",
              "passed": False, "upstream_commit": revision,
              "runner_sha256": sha(__file__), "runtime": runtime_identity(),
              "upstream_diff": subprocess.check_output([
                  "git", "-C", str(args.upstream), "diff", "HEAD"], text=True),
              "pool_config_sha256": sha(args.upstream / "vllm/v1/flexicache/config.json"),
              "model_config_sha256": sha(args.model / "config.json"),
              "data_sha256": sha(args.data_file), "task": "multifieldqa_en", "index": args.index}
    try:
        item = read_jsonl(args.data_file)[args.index]
        prompt = "[INST] " + DATASET_TO_PROMPT["multifieldqa_en"].format(**item) + " [/INST]"
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        ids = tokenizer(prompt).input_ids
        token_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        if token_hash != args.expected_token_hash:
            raise ValueError("Input tokens differ from the frozen control")
        if len(ids) > 31500:
            raise ValueError("Pilot requires an untruncated prompt")
        report.update({"sample_id": hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest(),
                       "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "input_tokens": len(ids), "input_token_sha256": token_hash})
        environment = {"VLLM_USE_V1": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                       "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1",
                       "CUDA_HOME": "/usr/local/cuda", "CUDACXX": "/usr/local/cuda/bin/nvcc",
                       "TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "8"}
        os.environ.update(environment)
        report["engine_environment"] = environment
        from vllm import LLM, SamplingParams
        import vllm
        if args.cudart_library is not None:
            from vllm.v1.worker import gpu_worker
            torch.cuda.init()
            report["runtime_resolver_adapter"] = pin_worker_cudart(gpu_worker, args.cudart_library)
        if args.l40s_prefill_stage_one:
            from flexicache_launch import StageOnePrefill
            from vllm.attention.ops import prefix_prefill
            adapter = StageOnePrefill(prefix_prefill.fwd_kernel_flexicache)
            prefix_prefill.fwd_kernel_flexicache = adapter
            report["prefill_launch_adapter"] = {
                "scope": "pipeline stages 3 to 1 only; kernel source and tiles unchanged",
                "adapter_sha256": sha(Path(__file__).with_name("flexicache_launch.py")),
                "kernel_source_sha256": sha(prefix_prefill.__file__), "calls": adapter.calls}
        if Path(vllm.__file__).resolve().parent != (args.upstream / "vllm").resolve():
            raise ValueError("Imported vLLM is not the frozen FlexiCache worktree")
        settings = dict(model=MODEL_ID, revision=REVISION, dtype="bfloat16", seed=0,
                        enable_flexicache=True, enable_prefix_caching=False,
                        disable_cascade_attn=True, gpu_memory_utilization=0.95,
                        max_num_seqs=1, max_model_len=31500, max_num_batched_tokens=2048,
                        num_unstable_heads=64, rerank_frequency=16, topK_budget=64,
                        unstable_heads_profile_task="gov_report", enforce_eager=True)
        report["engine_settings"] = settings
        report["vllm_version"] = vllm.__version__
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        llm = LLM(**settings)
        torch.cuda.synchronize()
        report["engine_initialization_s"] = time.perf_counter() - start
        report["engine_initialization_gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
        runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
        report["transfer_extensions"] = {name: {
            "path": getattr(runner, name).__file__,
            "sha256": sha(getattr(runner, name).__file__)} for name in
            ("_bt_tx", "_kv_d2h_tx", "_kv_h2d_tx", "_topk_swap_map")}
        report["allocated_pools"] = {name: pool_info(getattr(runner, name)) for name in
                                     ("gpu_kv_cache_pool", "cpu_kv_cache_pool", "minmax_key_cache")}
        for name in ("gpu_kv_cache_pool", "cpu_kv_cache_pool"):
            if getattr(runner, name).dtype != torch.bfloat16:
                raise ValueError("Expected bf16 GPU/CPU K/V pools")
        report["resource_observations"] = observe_resources(
            runner, llm.llm_engine.engine_core.engine_core)
        report["whole_request_timing_scope"] = "includes observation hooks; feasibility only"
        report["gpu_allocated_before_generation_bytes"] = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = llm.generate([{"prompt_token_ids": ids}], SamplingParams(
            temperature=0, top_p=1, seed=0, max_tokens=64), use_tqdm=False)[0]
        torch.cuda.synchronize()
        report["whole_request_s"] = time.perf_counter() - start
        if result.prompt_token_ids != ids:
            raise ValueError("Engine changed supplied prompt IDs")
        prediction = result.outputs[0].text
        report.update({"prediction": prediction,
                       "generated_tokens": len(result.outputs[0].token_ids),
                       "generated_token_ids": result.outputs[0].token_ids,
                       "score": 100 * max(qa_f1_score(prediction, answer) for answer in item["answers"]),
                       "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       "passed": True})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("passed", "input_tokens", "generated_tokens", "score")}))


if __name__ == "__main__":
    main()
