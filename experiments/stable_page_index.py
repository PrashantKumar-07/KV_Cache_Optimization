"""Experimental incremental centroids for stable slow-tier token pages."""
import torch

from longbench_eval import TieredKVCache


class StablePageIndex:
    def __init__(self, cache, page_size=16):
        self.page_size = page_size
        self.page_by_pos = {pos: i // page_size for i, pos in enumerate(cache.stt_pos)}
        self.next_page = (len(cache.stt_pos) + page_size - 1) // page_size
        self.next_fill = 0
        heads, rows, dim = cache.stt_k.shape
        self.sums = torch.zeros(heads, self.next_page, dim,
                                device=cache.stt_k.device, dtype=torch.float32)
        self.counts = torch.zeros(self.next_page, device=cache.stt_k.device,
                                  dtype=torch.long)
        live = torch.where(~cache.stt_shadow)[0]
        if live.numel():
            page_ids = (live // page_size).to(cache.stt_k.device)
            self.sums.index_add_(1, page_ids, cache.stt_k[:, live, :].float())
            self.counts.index_add_(0, page_ids, torch.ones_like(page_ids))
        self.changed_key_bytes = 0

    def _assign(self, pos):
        if pos not in self.page_by_pos:
            self.page_by_pos[pos] = self.next_page
            self.next_fill += 1
            if self.next_fill == self.page_size:
                self.next_page += 1
                self.next_fill = 0
        page = self.page_by_pos[pos]
        if page >= self.sums.shape[1]:
            extra = page + 1 - self.sums.shape[1]
            self.sums = torch.cat((self.sums, self.sums.new_zeros(
                self.sums.shape[0], extra, self.sums.shape[2])), dim=1)
            self.counts = torch.cat((self.counts, self.counts.new_zeros(extra)))
        return page

    def reconcile(self, before, cache):
        """Update only positions entering/leaving the live cold pool."""
        old_pos, old_shadow, old_k = before
        old_live = {p: i for i, p in enumerate(old_pos) if not old_shadow[i]}
        new_live = {p: i for i, p in enumerate(cache.stt_pos) if not cache.stt_shadow[i]}
        for old, new, keys, sign in ((old_live, new_live, old_k, -1),
                                     (new_live, old_live, cache.stt_k, 1)):
            changed = [(p, i) for p, i in old.items() if p not in new]
            if not changed:
                continue
            pages = torch.tensor([self._assign(p) for p, _ in changed],
                                 device=self.sums.device)
            indices = torch.tensor([i for _, i in changed], device=keys.device)
            values = keys.index_select(1, indices).float()
            self.sums.index_add_(1, pages, values * sign)
            self.counts.index_add_(0, pages, torch.full_like(pages, sign))
            self.changed_key_bytes += values.numel() * keys.element_size()
        if (self.counts < 0).any():
            raise RuntimeError('Negative live page count')

    def candidate_rows(self, q, cache, live, page_count, min_rows):
        """Return relative live-row indices; read page centers, then row metadata."""
        active = torch.where(self.counts > 0)[0]
        if not active.numel():
            return torch.empty(0, dtype=torch.long, device=cache.stt_k.device)
        centers = self.sums.index_select(1, active) / self.counts[active][None, :, None]
        scores = torch.einsum('ghd,gpd->ghp', q, centers).amax((0, 1))
        ranked = active[torch.argsort(scores, descending=True)].tolist()
        live_rows = live.tolist()
        by_page = {}
        for relative, row in enumerate(live_rows):
            page = self.page_by_pos[cache.stt_pos[row]]
            by_page.setdefault(page, []).append(relative)
        selected = []
        for rank, page in enumerate(ranked):
            selected.extend(by_page.get(page, []))
            if rank + 1 >= page_count and len(selected) >= min_rows:
                break
        return torch.tensor(selected, dtype=torch.long, device=cache.stt_k.device)

    def validate(self, cache):
        """Debug-only full rebuild to catch stale sums/counts."""
        live = torch.where(~cache.stt_shadow)[0]
        pages = torch.tensor([self.page_by_pos[cache.stt_pos[i]] for i in live.tolist()],
                             device=cache.stt_k.device)
        sums = torch.zeros_like(self.sums)
        counts = torch.zeros_like(self.counts)
        if live.numel():
            sums.index_add_(1, pages, cache.stt_k[:, live, :].float())
            counts.index_add_(0, pages, torch.ones_like(pages))
        if not torch.equal(counts, self.counts):
            raise AssertionError('Incremental page counts differ from full rebuild')
        torch.testing.assert_close(self.sums, sums, atol=1e-4, rtol=1e-5)


class IndexedTieredKVCache(TieredKVCache):
    """Opt-in cache instrumented with live-status delta updates."""
    def initial_bifurcation(self, *args, **kwargs):
        result = super().initial_bifurcation(*args, **kwargs)
        self.page_index = StablePageIndex(self)
        return result

    def _page_snapshot(self):
        return list(self.stt_pos), self.stt_shadow.clone(), self.stt_k

    def promote(self, hits):
        before = self._page_snapshot()
        result = super().promote(hits)
        self.page_index.reconcile(before, self)
        return result

    def evict_and_demote(self, new_k, new_v, new_pos):
        before = self._page_snapshot()
        result = super().evict_and_demote(new_k, new_v, new_pos)
        self.page_index.reconcile(before, self)
        return result
