"""Offline page-ranking diagnostics; summaries are rebuilt from full keys."""
import torch


def position_page_candidates(q, keys, positions, page_size, page_count):
    """Rank centroid pages keyed by absolute token position, not row index."""
    if page_size < 1:
        raise ValueError('Positive page size required')
    return group_page_candidates(q, keys, [p // page_size for p in positions], page_count)


def group_page_candidates(q, keys, group_ids, page_count):
    """Rank centroid pages with caller-maintained stable group IDs."""
    if page_count < 1 or keys.shape[1] < 1:
        raise ValueError('Positive page count and nonempty keys required')
    if len(group_ids) != keys.shape[1]:
        raise ValueError('One page ID required per key row')
    page_id = torch.as_tensor(group_ids, device=keys.device, dtype=torch.long)
    _, inverse = torch.unique(page_id, sorted=True, return_inverse=True)
    n_pages = int(inverse.max().item()) + 1
    centers = torch.zeros(keys.shape[0], n_pages, keys.shape[2],
                          device=keys.device, dtype=torch.float32)
    centers.index_add_(1, inverse, keys.float())
    counts = torch.bincount(inverse, minlength=n_pages)
    centers /= counts[None, :, None]
    scores = torch.einsum('ghd,gpd->ghp', q, centers)
    chosen = scores.amax((0, 1)).topk(min(page_count, n_pages)).indices
    rows = torch.where(torch.isin(inverse, chosen))[0]
    return rows, scores


def page_candidates(q, keys, page_size, page_count, bound_type='box'):
    """Return candidate row IDs and scores; q=[KV heads, groups, dim].

    Pages follow current row order. Box/L2 scores are upper bounds; centroid
    scores are heuristic. Rank by maximum query-head score.
    Summaries are rebuilt here: their maintenance cost is not optimized away.
    """
    if page_size < 1 or page_count < 1 or keys.shape[1] < 1:
        raise ValueError('Positive page size/count and nonempty keys required')
    if bound_type not in ('box', 'l2', 'centroid', 'centroid_loop'):
        raise ValueError('bound_type must be box, l2, centroid or centroid_loop')
    if bound_type == 'centroid':
        # One reduction per full-page tensor, rather than a Python/kernel
        # launch per page. Keep the last partial page's true row count.
        full, rem = divmod(keys.shape[1], page_size)
        centers = []
        if full:
            centers.append(keys[:, :full*page_size].float().reshape(
                keys.shape[0], full, page_size, keys.shape[2]).mean(dim=2))
        if rem:
            centers.append(keys[:, full*page_size:].float().mean(dim=1, keepdim=True))
        center = torch.cat(centers, dim=1)
        bounds = torch.einsum('ghd,gpd->ghp', q, center)
        chosen = bounds.amax((0, 1)).topk(min(page_count, bounds.shape[-1])).indices
        rows = (chosen[:, None]*page_size + torch.arange(page_size, device=keys.device)).flatten()
        return rows[rows < keys.shape[1]], bounds
    bounds = []
    for start in range(0, keys.shape[1], page_size):
        page = keys[:, start:start+page_size].float()
        if bound_type == 'box':
            lo, hi = page.amin(1), page.amax(1)
            bound = (q.clamp_min(0)*hi[:, None] + q.clamp_max(0)*lo[:, None]).sum(-1)
        else:
            center = page.mean(1)
            bound = (q * center[:, None]).sum(-1)
            if bound_type == 'l2':
                radius = (page - center[:, None]).norm(dim=-1).amax(1)
                bound = bound + q.norm(dim=-1) * radius[:, None]
        bounds.append(bound)
    bounds = torch.stack(bounds, -1)
    chosen = bounds.amax((0, 1)).topk(min(page_count, len(bounds[0, 0]))).indices
    rows = (chosen[:, None]*page_size + torch.arange(page_size, device=keys.device)).flatten()
    return rows[rows < keys.shape[1]], bounds


def check():
    torch.manual_seed(0)
    q, k = torch.randn(2, 3, 8), torch.randn(2, 19, 8)
    for bound_type in ('box', 'l2', 'centroid', 'centroid_loop'):
        for size in (1, 4, 32):
            rows, bounds = page_candidates(q, k, size, 2, bound_type)
            logits = torch.einsum('ghd,gnd->ghn', q, k)
            if bound_type in ('box', 'l2'):
                for i in range(bounds.shape[-1]):
                    assert (logits[..., i*size:(i+1)*size] <= bounds[..., i, None]+1e-5).all()
            assert len(rows.unique()) == len(rows) <= min(19, 2*size)
            all_rows, _ = page_candidates(q, k, size, 100, bound_type)
            assert sorted(all_rows.tolist()) == list(range(19))
    pos = [0, 1, 17, 18, 32, 33, 49, 51, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
    for size in (1, 4, 16):
        rows, scores = position_page_candidates(q, k, pos, size, 2)
        assert len(rows.unique()) == len(rows) <= 19
        all_rows, _ = position_page_candidates(q, k, pos, size, 100)
        assert all_rows.tolist() == list(range(19))
        assert scores.shape[-1] <= len(pos)
    print('PASS: box/L2 bounds and all-mode candidate coverage')


if __name__ == '__main__':
    check()
