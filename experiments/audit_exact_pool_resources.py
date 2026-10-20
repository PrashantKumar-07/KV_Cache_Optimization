"""Whole-decode Torch allocation envelope in the unchanged timing caller."""
import hashlib
from pathlib import Path
import torch
import compare_cpu_batched as trial
import compare_exact_pool as comparison
from audit_pinned_blocks import kernel


def main():
    original = trial.profile_one

    def profile(model, tokenizer, policy, *args, **kwargs):
        # Subtract only constant model storage, never temporary caller allocations.
        storages = {t.untyped_storage()._cdata: t.untyped_storage().nbytes()
                    for t in list(model.parameters()) + list(model.buffers())
                    if t.device.type == 'cuda'}
        weights = sum(storages.values())
        baseline = dict(allocated_bytes=torch.cuda.memory_allocated(),
                        reserved_bytes=torch.cuda.memory_reserved())
        result = original(model, tokenizer, policy, *args, **kwargs)
        root = kernel().block_info(policy.manager.pool)
        allocated_peak = torch.cuda.max_memory_allocated()
        reserved_peak = torch.cuda.max_memory_reserved()
        assert allocated_peak == round(result['peak_memory_gb'] * 1e9)
        result['whole_decode_resources'] = dict(
            constant_model_storage_bytes=weights, baseline=baseline,
            root=root, gpu_allocated_peak_bytes=allocated_peak,
            gpu_reserved_peak_bytes=reserved_peak,
            allocated_extra_plus_host_range_bytes=allocated_peak - weights + root['allocator_block_bytes'],
            reserved_extra_plus_host_range_bytes=reserved_peak - weights + root['allocator_block_bytes'],
            ceiling_bytes=512 * 2**20,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations='Independent maxima give an upper envelope for Torch-managed active allocation plus this host range. Reserved counts retained CUDA cache, including prefill blocks. Excludes CUDA-driver/non-Torch allocations and other host allocations/cache. Legacy timing caller releases prefill logits before decode; no timing claim for this diagnostic.')
        return result

    trial.profile_one = profile
    try:
        comparison.main()
    finally:
        trial.profile_one = original


if __name__ == '__main__':
    main()
