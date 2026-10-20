"""Storage-only native reuse lifetime and reused free-generation integrity gates."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import native_reuse_visible as native
from reuse_visible_policy import can_reuse
from fused_bounded_pool import FusedBoundedPoolPolicy
from check_batched_generation import Trajectory, measure
from provenance import atomic_json, read_jsonl, runtime_identity


class Control(Trajectory, FusedBoundedPoolPolicy):
    def __init__(self, **options):
        self.trajectory, self.frames = hashlib.sha256(), 0
        super().__init__(**options)


class Candidate(Trajectory, native.NativeReusePolicy):
    counts = {}

    def __init__(self, **options):
        self.trajectory, self.frames = hashlib.sha256(), 0
        super().__init__(**options)

    def reset(self):
        type(self).counts = dict(reused_layers=self.reused_layers,
            fallback_layers=self.reuse_fallback_layers,
            reused_compactions=sum(getattr(c, 'reused_compactions', 0) for c in self.layer_caches.values()))
        super().reset()


def native_checks():
    pattern = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(8,64,128)
    bits = torch.tensor([0,0x80000000,1,0x80000001,0x7f800000,0xff800000,
        0x7fc00001,0x7f800001], dtype=torch.int64).to(torch.int32).view(torch.float32).repeat(8)
    fields = ('sram_k', 'sram_v', 'sram_cum', 'sram_age')
    checks = []
    for keep in (list(range(64)), list(range(0,64,2)), [63,0,33,4,4]):
        states = [SimpleNamespace(sram_k=pattern.cuda(), sram_v=pattern.flip(0).cuda(),
            sram_cum=bits.cuda(), sram_age=bits.flip(0).cuda(), pending_keep=keep) for _ in range(4)]
        destinations = [(torch.full((1,8,96,128), -123, dtype=torch.bfloat16, device='cuda'),
            torch.full((1,8,96,128), -123, dtype=torch.bfloat16, device='cuda')) for _ in states]
        assert can_reuse(states, destinations, 32)
        index = torch.tensor(keep, device='cuda')
        expected = [{name:getattr(s,name).index_select(1 if name in fields[:2] else 0,index).cpu().contiguous().view(torch.uint8)
            for name in fields} for s in states]
        native.gather_into(states, destinations)
        for state, pair, values in zip(states, destinations, expected):
            for name in fields:
                assert torch.equal(getattr(state,name).cpu().contiguous().view(torch.uint8), values[name])
            for raw in pair:
                assert torch.all(raw[:,:,len(keep):,:] == -123)
            assert state.sram_k.untyped_storage()._cdata == pair[0].untyped_storage()._cdata
        checks.append(dict(case='all_bf16_bits_f32_payloads_tail_alias', kept_rows=len(keep), exact=True))
    producer = torch.cuda.Stream()
    consumers = [torch.cuda.Stream(), torch.cuda.Stream()]
    original = native.kernel
    module = original()
    class DelayedKernel:
        def compact(self, *args):
            torch.cuda._sleep(40000000)
            return module.compact(*args)
    native.kernel = lambda: DelayedKernel()
    try:
        for round in range(12):
            consumer = consumers[round % 2]
            with torch.cuda.stream(producer):
                states = [SimpleNamespace(sram_k=pattern.cuda(), sram_v=pattern.flip(0).cuda(),
                    sram_cum=bits.cuda(), sram_age=bits.flip(0).cuda(), pending_keep=list(range(64))) for _ in range(4)]
                destinations = [(torch.full((1,8,96,128),-123,dtype=torch.bfloat16,device='cuda'),
                    torch.full((1,8,96,128),-123,dtype=torch.bfloat16,device='cuda')) for _ in states]
                ready = torch.cuda.Event(); ready.record()
            with torch.cuda.stream(consumer):
                consumer.wait_event(ready)
                native.gather_into(states, destinations)
                snapshots = [{name:getattr(s,name).clone() for name in fields} for s in states]
                tails = [t[:,:,64:,:].clone() for pair in destinations for t in pair]
            # Release all source and destination owners while the consumer is delayed.
            del states, destinations
            with torch.cuda.stream(producer):
                pressure = [torch.full((8,64,128),-321,dtype=torch.bfloat16,device='cuda') for _ in range(32)]
                pressure += [torch.full((1,8,96,128),-321,dtype=torch.bfloat16,device='cuda') for _ in range(32)]
                pressure += [torch.full((64,),-321,dtype=torch.float32,device='cuda') for _ in range(32)]
            consumer.synchronize()
            for snapshot in snapshots:
                for name, expected in zip(fields,(pattern,pattern.flip(0),bits,bits.flip(0))):
                    assert torch.equal(snapshot[name].cpu().contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
            assert all(torch.all(t == -123).item() for t in tails)
            producer.synchronize()
            del pressure, snapshots, tails
            checks.append(dict(case='delayed_source_destination_release_allocator_pressure', round=round, consumer=round%2, exact=True))
    finally:
        native.kernel = original
        torch.cuda.synchronize()
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not args.native_only and args.manifest is None: parser.error('--manifest required for generation')
    sources = list(Path('src').glob('*.py')) + list(Path('experiments').glob('*.py')) + list(Path('experiments').glob('*.cu')) + list(Path('experiments').glob('*.cpp'))
    report = dict(passed=False, runtime=runtime_identity(), rows=[], native_checks=[],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    try:
        report['reuse_binary_sha256'] = hashlib.sha256(Path(native.kernel().__file__).read_bytes()).hexdigest()
        if args.native_only:
            report['native_checks'] = native_checks()
        else:
            m = json.loads(args.manifest.read_text())
            report.update(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(), hot=m['hot'], cold=m['cold'])
            requests = []
            for task, entry in m['tasks'].items():
                path = Path(entry['source_file'])
                assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['source_sha256']
                data = read_jsonl(path)
                for index, sample in zip(entry['selected_zero_based_indices'],entry['selected_sample_ids']):
                    item = data[index]
                    assert hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest() == sample
                    requests.append((task,sample,item))
            assert len(requests)==12 and m['seeds']==[0] and (m['hot'],m['cold']) in ((991,1090),(1024,2048))
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_pretrained(m['model'],torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
            tokenizer = AutoTokenizer.from_pretrained(m['model'])
            for number,(task,sample,item) in enumerate(requests):
                pair = {}
                for arm in (('fused_bounded','native_reuse') if number%2==0 else ('native_reuse','fused_bounded')):
                    result = measure(model,tokenizer,item,task,0,'cpu_scores',m['max_ctx'],prompt_queries=True,
                        policy_cls=Control if arm=='fused_bounded' else Candidate,
                        policy_options=dict(sram_budget=m['hot'],stt_budget=m['cold'],stt_expose_quota=32))
                    assert result['reserved_host_tensor_bytes']==(m['cold']+64)*131072
                    pair[arm] = result
                    report['rows'].append(dict(task=task,sample_id=sample,seed=0,arm=arm,
                        reuse_counts=Candidate.counts.copy() if arm=='native_reuse' else {},**result))
                    atomic_json(args.output,report)
                    print(number+1,task,arm,len(result['generated_token_ids']),flush=True)
                for field in ('prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256',
                    'state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes',
                    'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes','reserved_host_tensor_bytes'):
                    assert pair['fused_bounded'][field]==pair['native_reuse'][field],(sample,field)
            assert len(report['rows'])==24
        report.update(passed=True,limitations='Reused development samples and one seed; exact preservation only. Native delayed explicit-event two-consumer tests release both input and destination owners; serialized model execution, no threaded serving/quality-superiority/physical STT claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output,report)


if __name__=='__main__': main()
