"""Observe known live KV storage at fused launch and every decode boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import audit_snap_tier_envelope as envelope
from audit_snap_tier_resources import distinct_storage_bytes
import fused_hot_compaction as compaction
from fused_bounded_pool import FusedBoundedPoolPolicy
from provenance import atomic_json, runtime_identity


class ObservedPolicy(FusedBoundedPoolPolicy):
    launches=[]
    active=None

    def _visible_tuple(self,past):
        type(self).active=(self,past)
        try:
            return super()._visible_tuple(past)
        finally:
            type(self).active=None


class ObservedKernel:
    def __init__(self,module):
        self.module=module

    def compact(self,pointers,indices,sizes,kv,metadata):
        policy,past=ObservedPolicy.active
        caches=list(policy.layer_caches.values())
        roles=dict(pinned_pool=[policy.manager.pool],raw_visible=[t for pair in past for t in pair],
            hot_inputs=[t for c in caches for t in (c.sram_k,c.sram_v)],
            queued_d2h=[packed for _,packed,_ in policy.manager.pending],fused_output=[kv],
            prefetched=[t for c in caches if c._prefetched is not None for t in c._prefetched[1]])
        tensors=[t for group in roles.values() for t in group]
        unique=distinct_storage_bytes(tensors)
        ObservedPolicy.launches.append(dict(step=policy.decode_step_count,
            known_live_kv_storage_bytes=unique,known_live_kv_total_bytes=sum(unique.values()),
            role_storage_bytes={name:distinct_storage_bytes(group) for name,group in roles.items()},
            source_rows=sizes.cpu().tolist(),kept_rows=indices.shape[1],
            output_kv_bytes=kv.untyped_storage().nbytes(),output_metadata_bytes=metadata.untyped_storage().nbytes(),
            process_gpu_allocated_bytes=torch.cuda.memory_allocated(),
            process_gpu_reserved_bytes=torch.cuda.memory_reserved()))
        return self.module.compact(pointers,indices,sizes,kv,metadata)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    files=list(Path('experiments').glob('*.py'))+list(Path('src').glob('*.py'))+[
        Path('experiments/native_hot_compact.cpp'),Path('experiments/native_hot_compact.cu')]
    report=dict(passed=False,runtime=runtime_identity(),settings=envelope.SETTINGS,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        boundary_kv_ceiling_bytes=envelope.KV_CEILING,forward_width_ceiling=envelope.WIDTH_CEILING)
    original_cls,original_kernel=envelope.DirectScorePolicy,compaction.kernel
    try:
        module=original_kernel()
        report['binary_sha256']=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        envelope.DirectScorePolicy=ObservedPolicy
        compaction.kernel=lambda:ObservedKernel(module)
        ObservedPolicy.launches=[]
        prompt_path=Path('results/remediation_v3/latency_prompt.txt')
        teacher_path=Path('results/shadow_ablation_full/fixed_stream_20260929.json')
        stream=json.loads(teacher_path.read_text())['config']['teacher_token_ids'][:50]
        assert len(stream)==50
        torch.manual_seed(0)
        model_path='../models/mistral-7b-instruct'
        model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(model_path)
        ids=tokenizer(prompt_path.read_text(),return_tensors='pt').input_ids[:,:16384].to('cuda')
        assert ids.shape==(1,16384)
        report.update(model=model_path,seed=0,context=16384,decode=50,
            gpu=torch.cuda.get_device_name(),prompt_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
            teacher_source_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
            input_token_ids=ids.cpu().tolist()[0],teacher_token_ids=stream)
        report['result']=envelope.run(model,ids,stream,'cpu_direct')
        report['gather_launches']=ObservedPolicy.launches
        assert len(ObservedPolicy.launches)==50
        old=json.loads(Path('results/baseline_fidelity/snap_tier_owned_envelope_20261002/result.json').read_text())
        reference=next(r for r in old['rows'] if r['arm']=='cpu_direct')
        for field in ('predicted_token_ids','counters','h2d_payload_bytes','d2h_payload_bytes'):
            assert report['result'][field]==reference[field],field
        report.update(passed=True,policy_preserved=True,
            maximum_known_gather_kv_bytes=max(r['known_live_kv_total_bytes'] for r in ObservedPolicy.launches),
            known_gather_storage_within_512mib=all(r['known_live_kv_total_bytes']<=envelope.KV_CEILING
                for r in ObservedPolicy.launches),
            limitations='Diagnostic CUDA synchronization outside timing. Known live KV at gather launch includes raw visible tensors, old hot inputs, new fused output, pinned root and queued/prefetched transfers. Metadata, unrelated Python locals, attention/model temporaries and retained allocator blocks are excluded; this is a lower bound on peak live storage, not full physical memory equality. Frozen capacities and teacher stream, no retuning.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        envelope.DirectScorePolicy=original_cls
        compaction.kernel=original_kernel
        ObservedPolicy.active=None
        atomic_json(args.output,report)


if __name__=='__main__':main()
