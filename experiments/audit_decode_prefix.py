"""Untimed bit/ownership/capacity audit before ordinary bulk promotion."""
import hashlib
from pathlib import Path
import torch
import compare_bulk_promotion as comparison


def bits_hash(tensor):
    return hashlib.sha256(tensor.detach().view(torch.int16).cpu().contiguous().numpy().tobytes()).hexdigest()


class ObservedBulk(comparison.RecordedBulkPolicy):
    observations = []

    def _prepare_decode_reads(self):
        past = self.incoming_past
        first = next(iter(self.layer_caches))
        n_new = max(1, past[0][0].shape[2] - self._vis_len[first])
        checks, weak = [], []
        try:
            for layer, cache in self.layer_caches.items():
                key, value = past[layer]
                hot = len(cache.sram_pos)
                equal = all(torch.equal(raw[0, :, :hot].view(torch.int16), old.view(torch.int16))
                            for raw, old in ((key, cache.sram_k), (value, cache.sram_v)))
                row = dict(layer=layer, hot_before=hot, raw_rows=key.shape[2], new_rows=n_new,
                    prefix_bits_equal=equal,
                    storage_distinct=all(raw.untyped_storage()._cdata != old.untyped_storage()._cdata
                                         for raw, old in ((key, cache.sram_k), (value, cache.sram_v))),
                    raw_key_stride=list(key.stride()), hot_key_stride=list(cache.sram_k.stride()),
                    raw_storage_bytes=key.untyped_storage().nbytes()+value.untyped_storage().nbytes(),
                    hot_storage_bytes=cache.sram_k.untyped_storage().nbytes()+cache.sram_v.untyped_storage().nbytes(),
                    new_tail_hashes=[bits_hash(raw[:, :, -n_new:]) for raw in (key,value)])
                assert equal, ('incoming hot prefix',layer)
                assert row['storage_distinct'], ('old hot ownership',layer)
                handles = [old.untyped_storage()._weak_ref() for old in (cache.sram_k,cache.sram_v)]
                weak.extend(handles)
                checks.append((row,handles))
            result = super()._prepare_decode_reads()
            for row, handles in checks:
                layer=row['layer']; cache=self.layer_caches[layer]
                promoted = len(cache.sram_pos)-row['hot_before']
                row.update(promoted_rows=promoted,
                    extended_rows=len(cache.sram_pos)+n_new,
                    capacity_fits=len(cache.sram_pos)+n_new<=row['raw_rows'],
                    old_hot_expired_after_preparation=[torch.UntypedStorage._expired(h) for h in handles],
                    new_tail_unchanged=row['new_tail_hashes']==[
                        bits_hash(raw[:, :, -n_new:]) for raw in past[layer]])
                assert row['capacity_fits'] and row['new_tail_unchanged'],row
                type(self).observations.append(dict(prompt_tokens=cache.metrics.prompt_len,
                    step=self.decode_step_count,**row))
            return result
        finally:
            for handle in weak: torch.UntypedStorage._free_weak_ref(handle)


def main():
    previous, writer = comparison.RecordedBulkPolicy, comparison.driver.atomic_json
    ObservedBulk.observations=[]
    def write(path,report):
        report.update(prefix_audit=True, prefix_observations=ObservedBulk.observations,
            prefix_audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            prefix_audit_limitations='Before/after read-only observer; GPU equality sync and CPU bit hashes invalidate timing/memory effects. Weak handles retain no GPU payload. Capacity/prefix facts are necessary, not sufficient proof of safe in-place compaction or ordinary generation/serving reuse. No candidate/default change.')
        writer(path,report)
    comparison.RecordedBulkPolicy, comparison.driver.atomic_json=ObservedBulk,write
    try: comparison.main()
    finally: comparison.RecordedBulkPolicy, comparison.driver.atomic_json=previous,writer


if __name__=='__main__': main()
