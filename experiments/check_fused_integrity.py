"""Native bit/stream lifetime and six-task free-generation preservation."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import fused_hot_compaction as compaction
from fused_eviction_policy import FusedEvictionPolicy
from fused_bounded_pool import FusedBoundedPoolPolicy
from check_batched_generation import Trajectory,measure
from check_bounded_pool_generation import Control
from provenance import atomic_json,read_jsonl,runtime_identity


class Candidate(Trajectory,FusedEvictionPolicy):
    counts={}

    def __init__(self,**options):
        self.trajectory,self.frames=hashlib.sha256(),0
        super().__init__(**options)

    def reset(self):
        type(self).counts={name:sum(getattr(c,name,0) for c in self.layer_caches.values())
                          for name in ('fused_compactions','eager_compactions')}
        super().reset()


class BoundedCandidate(Candidate,FusedBoundedPoolPolicy):
    """Same observer and reset accounting over the composed policy."""


def native_checks(module):
    checks=[]
    pattern=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(8,64,128)
    bits=torch.tensor([0,0x80000000,1,0x80000001,0x7f800000,0xff800000,
        0x7fc00001,0x7f800001],dtype=torch.int64).to(torch.int32).view(torch.float32).repeat(8)
    for keep in (list(range(64)),list(range(0,64,2)),[63,0,33,4,4]):
        states=[SimpleNamespace(sram_k=pattern.to('cuda'),sram_v=pattern.flip(0).to('cuda'),
            sram_cum=bits.to('cuda'),sram_age=bits.flip(0).to('cuda')) for _ in range(4)]
        indices=torch.tensor(keep,device='cuda')
        expected=[{name:getattr(s,name).index_select(1 if name in ('sram_k','sram_v') else 0,indices).cpu().contiguous().view(torch.uint8)
            for name in ('sram_k','sram_v','sram_cum','sram_age')} for s in states]
        compaction.gather(states,[keep]*4)
        for state,values in zip(states,expected):
            for name,value in values.items():
                assert torch.equal(getattr(state,name).cpu().contiguous().view(torch.uint8),value)
        checks.append(dict(case='all_bf16_bitpatterns_f32_signedzero_subnormal_nan',kept_rows=len(keep),layers=4,exact=True))
    producer=torch.cuda.Stream()
    consumers=[torch.cuda.Stream(),torch.cuda.Stream()]
    original=compaction.kernel
    class DelayedKernel:
        def compact(self,*args):
            torch.cuda._sleep(40000000)
            return module.compact(*args)
    compaction.kernel=lambda:DelayedKernel()
    try:
        for round in range(12):
            consumer=consumers[round%2]
            with torch.cuda.stream(producer):
                states=[SimpleNamespace(sram_k=pattern.to('cuda'),sram_v=pattern.flip(0).to('cuda'),
                    sram_cum=bits.to('cuda'),sram_age=bits.flip(0).to('cuda')) for _ in range(4)]
                ready=torch.cuda.Event();ready.record()
            with torch.cuda.stream(consumer):
                consumer.wait_event(ready)
                compaction.gather(states,[list(range(64))]*4)
            # Old fields were replaced and their last Python references released.
            # Try reusing producer allocations while the custom gather is delayed.
            with torch.cuda.stream(producer):
                pressure=[torch.full((8,64,128),-321,dtype=torch.bfloat16,device='cuda') for _ in range(32)]
                pressure.extend(torch.full((64,),-123,dtype=torch.float32,device='cuda') for _ in range(32))
            consumer.synchronize()
            for state in states:
                for name,expected in (('sram_k',pattern),('sram_v',pattern.flip(0)),('sram_cum',bits),('sram_age',bits.flip(0))):
                    assert torch.equal(getattr(state,name).cpu().contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
            producer.synchronize()
            del states,pressure
            checks.append(dict(case='delayed_gather_input_release_allocator_pressure',round=round,consumer=round%2,exact=True))
    finally:
        compaction.kernel=original
        torch.cuda.synchronize()
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--bounded-composition',action='store_true')
    args=parser.parse_args();base=args.directory
    output=base/'generation.json'
    if output.exists():raise FileExistsError(output)
    manifest_path=base/'generation_manifest.json'
    m=json.loads(manifest_path.read_text())
    sources=list(Path('src').glob('*.py'))+list(Path('experiments').glob('*.py'))+[
        Path('experiments/native_hot_compact.cpp'),Path('experiments/native_hot_compact.cu')]
    report=dict(passed=False,bounded_composition=args.bounded_composition,runtime=runtime_identity(),rows=[],native_checks=[],
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    try:
        module=compaction.kernel()
        report['binary_sha256']=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if args.bounded_composition:
            previous=Path('results/hardware/fused_eviction_stress_20261002/verified_report.json')
            proof=json.loads(previous.read_text())
            assert proof['verified'] and proof['native_bit_and_stream_lifetime_checks']==15
            report['unchanged_native_proof_sha256']=hashlib.sha256(previous.read_bytes()).hexdigest()
        else:
            report['native_checks']=native_checks(module)
        atomic_json(output,report)
        requests=[]
        for task,entry in m['tasks'].items():
            path=Path(entry['source_file'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['source_sha256']
            data=read_jsonl(path)
            for index,sample in zip(entry['selected_zero_based_indices'],entry['selected_sample_ids']):
                item=data[index]
                assert hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()==sample
                requests.append((task,sample,item))
        assert len(requests)==12 and m['seeds']==[0] and (m['hot'],m['cold']) in ((1024,2048),(991,1090))
        candidate_cls=BoundedCandidate if args.bounded_composition else Candidate
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained(m['model'],torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(m['model'])
        report['gpu']=torch.cuda.get_device_name()
        for index,(task,sample,item) in enumerate(requests):
            pair={}
            for arm in (('direct','fused') if index%2==0 else ('fused','direct')):
                result=measure(model,tokenizer,item,task,0,'cpu_scores',31500,prompt_queries=True,
                    policy_cls=Control if arm=='direct' else candidate_cls,
                    policy_options=dict(sram_budget=m['hot'],stt_budget=m['cold'],stt_expose_quota=32))
                slots=m['cold']+64 if arm=='fused' and args.bounded_composition else m['cold']+m['hot']+1
                assert result['reserved_host_tensor_bytes']==slots*131072
                pair[arm]=result
                report['rows'].append(dict(task=task,sample_id=sample,seed=0,arm=arm,
                    compaction_counts=candidate_cls.counts if arm=='fused' else {},**result))
                atomic_json(output,report)
                print(index+1,task,arm,len(result['generated_token_ids']),flush=True)
            for field in ('prediction','generated_token_ids','score','generation','max_gen','termination',
                'prompt_sha256','state_sha256','trajectory_sha256','trajectory_frames','counters',
                'prefill_query_shapes','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes'):
                assert pair['direct'][field]==pair['fused'][field],(sample,field)
        assert len(report['rows'])==24
        report.update(passed=True,limitations='Reused12developmentdocuments/sixfixedtasks/onegreedyseed/nativecaps+EOS; exactpolicy preservation, no qualitysuperiority. Native delayedtwo-consumer/source-producer-stream ownership test uses explicit producer-event waits and allocationpressure; no asynchronousmodelserving/threads/throughput/physicalSTT claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:atomic_json(output,report)


if __name__=='__main__':main()
