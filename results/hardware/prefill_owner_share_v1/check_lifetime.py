"""Untimed GPU ownership/bit checks with explicit caller-managed stream handoff."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments'))
from forward_owner_release import OwnerRelease
from prefill_owner_share import share_prefill_hot
from raw_input_reuse import gather_in_place
from provenance import atomic_json, runtime_identity

FIELDS = ('sram_k', 'sram_v', 'sram_cum', 'sram_age')
bits = lambda t: t.detach().cpu().contiguous().view(torch.uint8)


def release_handles(handles):
    for handle in handles:
        torch.UntypedStorage._free_weak_ref(handle)


def record_inputs(policy, visible, stream):
    # Cross-stream handoff is the caller's responsibility, not a prototype claim.
    for pair in visible:
        for tensor in pair:
            tensor.record_stream(stream)
    for cache in policy.layer_caches.values():
        for name in FIELDS:
            getattr(cache, name).record_stream(stream)


def gather_case(producer, consumer, keep):
    with torch.cuda.stream(producer):
        pattern = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1, 8, 64, 128)
        payload = torch.tensor([0, 0x80000000, 1, 0x80000001, 0x7f800000,
            0xff800000, 0x7fc00001, 0x7f800001], dtype=torch.int64).to(torch.int32).view(torch.float32).repeat(8)
        visible = []; caches = {}
        for layer in range(2):
            pair = tuple(torch.cat((value.cuda(), torch.full((1, 8, 32, 128), -123,
                dtype=torch.bfloat16, device='cuda')), dim=2) for value in (pattern, pattern.flip(1)))
            visible.append(pair)
            caches[layer] = SimpleNamespace(sram_pos=list(range(64)),
                sram_k=pair[0][0, :, :64].clone(), sram_v=pair[1][0, :, :64].clone(),
                sram_cum=payload.cuda(), sram_age=payload.flip(0).cuda(), pending_keep=keep)
        policy = SimpleNamespace(layer_caches=caches)
        ready = torch.cuda.Event(); ready.record()
    handles = [getattr(c, name).untyped_storage()._weak_ref()
               for c in caches.values() for name in FIELDS[:2]]
    try:
        with torch.cuda.stream(consumer):
            consumer.wait_event(ready)
            record_inputs(policy, visible, consumer)
            expected = [{name: getattr(c, name).cpu().index_select(
                1 if name in FIELDS[:2] else 0, torch.tensor(keep, dtype=torch.long)).contiguous().view(torch.uint8)
                for name in FIELDS} for c in caches.values()]
            tails = [bits(t[:, :, len(keep):]) for pair in visible for t in pair]
            assert share_prefill_hot(policy, visible) == 2
            assert all(torch.UntypedStorage._expired(h) for h in handles)
            torch.cuda._sleep(40000000)
            for cache, pair in zip(caches.values(), visible):
                cache._hot_buffer = pair
                gather_in_place(cache)
            snapshots = [{name: getattr(c, name).clone() for name in FIELDS} for c in caches.values()]
            tail_snapshots = [t[:, :, len(keep):].clone() for pair in visible for t in pair]
        # Release all input/destination owners while their recorded consumer is delayed.
        del cache, pair, caches, visible, policy
        with torch.cuda.stream(producer):
            pressure = [torch.full((1, 8, 96, 128), -321, dtype=torch.bfloat16, device='cuda') for _ in range(32)]
            pressure += [torch.full((64,), -321, dtype=torch.float32, device='cuda') for _ in range(32)]
        consumer.synchronize()
        for snapshot, want in zip(snapshots, expected):
            for name in FIELDS:
                assert torch.equal(bits(snapshot[name]), want[name]), name
        assert all(torch.equal(bits(actual), want) for actual, want in zip(tail_snapshots, tails))
        producer.synchronize()
        del pressure
        return dict(case='all_bf16_f32_bits_overlapping_gather_release_pressure', kept_rows=len(keep),
            old_hot_owners_expired=True, exact_payload_and_untouched_tail=True)
    finally:
        release_handles(handles)


def owner_case(producer, consumer, fail, fixture_module):
    policy, past = fixture_module.fixture()
    with torch.cuda.stream(producer):
        for entry in past.layers:
            entry.keys, entry.values = entry.keys.cuda(), entry.values.cuda()
        for cache in policy.layer_caches.values():
            for name in FIELDS:
                setattr(cache, name, getattr(cache, name).cuda())
        ready = torch.cuda.Event(); ready.record()
    model = fixture_module.Model(policy, fail); lease = OwnerRelease(model, policy)
    handles = []
    try:
        with torch.cuda.stream(consumer):
            consumer.wait_event(ready)
            record_inputs(policy, [(l.keys, l.values) for l in past.layers], consumer)
            expected = [(bits(c.sram_k), bits(c.sram_v)) for c in policy.layer_caches.values()]
            assert share_prefill_hot(policy, tuple((l.keys, l.values) for l in past.layers)) == 2
            handles = [t.untyped_storage()._weak_ref() for l in past.layers for t in (l.keys, l.values)]
            try:
                model(torch.tensor([[1]], device='cuda'), past_key_values=past, use_cache=True)
            except RuntimeError as error:
                assert fail and str(error) == 'injected partial forward'
            else:
                assert not fail
            assert model.saw_released and lease.pending is None and policy.lease_restored
            assert lease.released_forwards == lease.restored_forwards == 1 and lease.released_layers == 2
            for layer, (cache, pair) in enumerate(zip(policy.layer_caches.values(), expected)):
                assert torch.equal(bits(cache.sram_k), pair[0]) and torch.equal(bits(cache.sram_v), pair[1])
                assert cache.sram_k.data_ptr() == past.layers[layer].keys.data_ptr()
            expiry = [torch.UntypedStorage._expired(h) for h in handles]
            assert expiry == ([True, True, False, False] if fail else [True] * 4)
        lease.close()
        assert not model._forward_pre_hooks and not model._forward_hooks
        return dict(case='gpu_partial_exception_restoration' if fail else 'gpu_forward_restoration',
            exact_payload=True, balanced_lease=True, expected_old_model_owner_expiry=expiry, hooks_removed=True)
    finally:
        lease.close(); release_handles(handles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = ROOT / 'tests/test_forward_owner_release.py'
    spec = importlib.util.spec_from_file_location('ownership_fixture', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    names = ('experiments/prefill_owner_share.py', 'experiments/forward_owner_release.py',
        'experiments/raw_input_reuse.py', 'tests/test_forward_owner_release.py')
    report = dict(passed=False, runtime=runtime_identity(), checks=[],
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits='Untimed helper/real Torch hook checks; toy DynamicCache model, not Mistral compute. '
            'Producer-to-consumer handoff uses explicit events and caller record_stream. '
            'No automatic cross-stream ownership, concurrent model serving, latency or ordinary adoption claim.')
    try:
        producer = torch.cuda.Stream(); consumers = [torch.cuda.Stream(), torch.cuda.Stream()]
        for number, consumer in enumerate(consumers):
            for keep in (list(range(64)), list(range(0, 64, 2)), [63, 0, 33, 4, 4], [], [63]):
                report['checks'].append(dict(consumer=number, **gather_case(producer, consumer, keep)))
                atomic_json(args.output, report)
            for fail in (False, True):
                report['checks'].append(dict(consumer=number, **owner_case(producer, consumer, fail, module)))
                atomic_json(args.output, report)
        assert len(report['checks']) == 14
        torch.cuda.synchronize(); report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.output, report)
    print('14 GPU bit/owner/exception checks pass', flush=True)


if __name__ == '__main__':
    main()
