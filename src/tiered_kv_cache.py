"""
tiered_kv_cache.py
------------------
The core contribution: a 3-tier KV-cache memory hierarchy with query-aware
routing. Pure PyTorch, hardware-agnostic, instrumented end to end.

Tier 1  GPU_VRAM - hot working set (sinks + recent window + hot heavy-hitters)
Tier 2  STT-RAM  - warm victim cache (simulated)
Tier 3  DRAM     - cold storage / permanent eviction

Tensor convention: (num_heads H, seq, head_dim D).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from attention import scaled_dot_product_attention, snapkv_importance
from cost_model import CostModel
from metrics import StepRecord, RunMetrics


@dataclass
class TieredConfig:
    num_heads: int = 8
    head_dim: int = 64
    sink_size: int = 4           # attention sinks pinned to VRAM forever
    window_size: int = 16        # recent sliding window pinned to VRAM
    sram_capacity: int = 64      # max tokens in VRAM (Tier 1) - kept for compat
    sttram_capacity: int = 128   # max tokens in STT-RAM (Tier 2)
    page_size: int = 16          # Quest sketch granularity
    promote_top_pages: int = 2   # max pages promoted per step
    store_dram: bool = False     # True: keep cold tokens in DRAM; False: drop
    pool_kernel: int = 5         # SnapKV clustering pool
    inclusive: bool = True       # inclusive victim cache for write savings
    # Fraction of sttram_capacity that initial_bifurcation is allowed to
    # fill with COLD PROMPT tokens (the ones that lost the SnapKV-importance
    # cut for VRAM). Default 1.0 preserves the original behavior: bifurcate
    # fills STT-RAM to capacity immediately whenever the prompt has more
    # spillover than sttram_capacity (true for any document longer than the
    # VRAM+STT budget -- i.e. most real long-context documents).
    #
    # Measured consequence of the default on a real 6.7k-token LongBench
    # sample at budget=1024: STT-RAM starts at 100% capacity from step 0,
    # so EVERY decode-time demotion immediately forces an STT overflow, and
    # "reclaim shadow backups first" (the safe, non-lossy overflow policy —
    # see evict_and_demote) throws away a just-created backup before its
    # promoted twin can ever be re-demoted into it. Result: write_savings_pct
    # measured at 0.0% end to end, regardless of promotion rate or eviction
    # scoring -- not a bug in either of those, but Tier 2 having zero spare
    # capacity to play its "victim cache" role at all once it is pre-filled
    # as a residual prompt archive.
    #
    # Lowering this value reserves headroom in STT-RAM for decode-time
    # backups (fewer cold prompt tokens survive bifurcation; more are
    # dropped outright), trading recall of prompt content chosen at
    # bifurcation time for the ability to demonstrate write savings during
    # decode. Where to land on that trade is a paper-level decision, not
    # one this code should make silently -- left at 1.0 (no behavior
    # change) until set explicitly.
    sttram_bifurcation_frac: float = 1.0
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


class TieredKVCache:
    def __init__(self, cfg: TieredConfig, cost_model: CostModel | None = None):
        self.cfg = cfg
        self.cm = cost_model or CostModel(bytes_per_elem=2)
        H, D = cfg.num_heads, cfg.head_dim
        dev, dt = cfg.device, cfg.dtype

        # ---- Tier 1: VRAM (hot working set) ----
        self.sram_k = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.sram_v = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.sram_pos = []                                     # original token index
        self.sram_cum = torch.empty(0, device=dev, dtype=dt)   # cumulative attn
        self.sram_age = torch.empty(0, device=dev, dtype=dt)   # # compute_attention() passes seen
        self.sram_grace = []    # True = promoted this step, protected from eviction
                                 # until it has survived one compute_attention() pass

        # ---- Tier 2: STT-RAM ----
        self.stt_k = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.stt_v = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.stt_pos = []
        self.stt_last_access = torch.empty(0, device=dev, dtype=dt)
        self.stt_shadow = []  # True = also in VRAM (inclusive backup)
        # peek/promote hysteresis (see resolve_promotions docstring):
        self.stt_was_vram_resident = []   # True = has left VRAM before (via
                                           # promote+evict); re-promotion of
                                           # these is gated on repeated
                                           # candidacy instead of the first hit
        self.stt_peek_streak = torch.empty(0, device=dev, dtype=dt)  # consecutive
                                           # steps this token has cleared the
                                           # sketch bar (resets to 0 otherwise)

        # ---- Tier 3: DRAM (optional) ----
        self.dram_k = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.dram_v = torch.empty(H, 0, D, device=dev, dtype=dt)
        self.dram_pos = []

        self.metrics = RunMetrics(label="TieredKV", num_heads=H, head_dim=D)
        self._t = 0  # logical clock for LRU

    # =====================================================================
    # PHASE 0 : initial bifurcation of the prompt
    # =====================================================================
    def initial_bifurcation(self, k_all, v_all, q_all):
        """
        k_all, v_all, q_all : (H, N, D) for the prompt.
        Routes prompt tokens into three tiers using SnapKV importance.
        """
        cfg = self.cfg
        H, N, D = k_all.shape
        self.metrics.prompt_len = N

        w = min(cfg.window_size, N)
        q_window = q_all[:, N - w:, :]
        importance = snapkv_importance(q_window, k_all, cfg.pool_kernel)

        sink_idx = list(range(min(cfg.sink_size, N)))
        window_idx = list(range(max(0, N - w), N))
        pinned = sorted(set(sink_idx) | set(window_idx))

        rest = [i for i in range(N) if i not in set(pinned)]
        rest_sorted = sorted(rest, key=lambda i: importance[i].item(), reverse=True)

        sram_budget = max(0, cfg.sram_capacity - len(pinned))
        sram_extra = rest_sorted[:sram_budget]
        after_sram = rest_sorted[sram_budget:]

        stt_fill_cap = int(cfg.sttram_capacity * cfg.sttram_bifurcation_frac)
        stt_sel = after_sram[:stt_fill_cap]
        cold_sel = after_sram[stt_fill_cap:]

        sram_idx = sorted(set(pinned) | set(sram_extra))
        stt_idx = sorted(stt_sel)
        cold_idx = sorted(cold_sel)

        self.sram_k = k_all[:, sram_idx, :].clone()
        self.sram_v = v_all[:, sram_idx, :].clone()
        self.sram_pos = list(sram_idx)
        self.sram_cum = importance[sram_idx].clone().to(self.cfg.dtype)
        self.sram_age = torch.zeros(len(sram_idx), device=cfg.device, dtype=cfg.dtype)
        self.sram_grace = [False] * len(sram_idx)

        self.stt_k = k_all[:, stt_idx, :].clone()
        self.stt_v = v_all[:, stt_idx, :].clone()
        self.stt_pos = list(stt_idx)
        self.stt_last_access = torch.zeros(len(stt_idx), device=cfg.device, dtype=cfg.dtype)
        self.stt_shadow = [False] * len(stt_idx)
        # Prompt tokens routed straight to STT-RAM at bifurcation have never
        # been VRAM-resident, so they promote on first candidacy just like
        # before peek/promote existed -- gating only applies to re-entry.
        self.stt_was_vram_resident = [False] * len(stt_idx)
        self.stt_peek_streak = torch.zeros(len(stt_idx), device=cfg.device, dtype=cfg.dtype)

        if cfg.store_dram:
            self.dram_k = k_all[:, cold_idx, :].clone()
            self.dram_v = v_all[:, cold_idx, :].clone()
            self.dram_pos = list(cold_idx)

        return {
            "sram": len(sram_idx), "sttram": len(stt_idx),
            "dram_or_dropped": len(cold_idx), "prompt_len": N,
        }

    # =====================================================================
    # STEP 1 : sketch check (Quest min/max upper bound)
    # =====================================================================
    def _page_bounds(self):
        """Return per-page (min_k, max_k) sketches and token-index slices.

        Fully vectorized: computes min/max for every page in ONE batched
        reduction instead of one GPU call per page (measured to be a
        meaningful share of sketch_check's cost with dozens of pages,
        called every decode step for every layer).
        """
        H, m, D = self.stt_k.shape
        ps = self.cfg.page_size
        if m == 0:
            return []

        num_pages = (m + ps - 1) // ps
        pad = num_pages * ps - m
        dev = self.stt_k.device

        shadow_t = torch.tensor(self.stt_shadow, dtype=torch.bool, device=dev)
        if pad > 0:
            k_pad = torch.cat(
                [self.stt_k, torch.zeros(H, pad, D, device=dev, dtype=self.stt_k.dtype)], dim=1
            )
            shadow_pad = torch.cat(
                [shadow_t, torch.ones(pad, dtype=torch.bool, device=dev)]  # padding excluded like a shadow row
            )
        else:
            k_pad, shadow_pad = self.stt_k, shadow_t

        k_reshaped = k_pad.view(H, num_pages, ps, D)
        shadow_reshaped = shadow_pad.view(num_pages, ps)
        mask = shadow_reshaped.view(1, num_pages, ps, 1)

        mins = k_reshaped.masked_fill(mask, float("inf")).min(dim=2).values   # (H, num_pages, D)
        maxs = k_reshaped.masked_fill(mask, float("-inf")).max(dim=2).values  # (H, num_pages, D)
        valid_page = (~shadow_reshaped).any(dim=1)  # (num_pages,)

        pages = []
        for pi in torch.where(valid_page)[0].tolist():
            start, end = pi * ps, min((pi + 1) * ps, m)
            pages.append((start, end, mins[:, pi, :], maxs[:, pi, :]))
        return pages

    def sketch_check(self, q):
        """
        q : (H, 1, D). Returns list of (start, end) slices that clear the
        cheap sketch bound -- i.e. PEEK/PROMOTION CANDIDATES, not
        automatically promotions. resolve_promotions() below decides which
        of these actually get physically moved to VRAM this step versus
        merely attended to in place (see its docstring). Also returns the
        sketch MAC count.
        """
        H, m, D = self.stt_k.shape
        if m == 0:
            return [], 0

        pages = self._page_bounds()
        if not pages:
            return [], 0
        q2 = q.squeeze(1)  # (H, D)

        # Batch all pages into one (P, H, D) stack so dmin/dmax/ub are one
        # vectorized op each instead of one Python-level `.item()` sync per
        # page (up to ~40+ pages/call at realistic STT-RAM sizes).
        mn_stack = torch.stack([p[2] for p in pages], dim=0)  # (P, H, D)
        mx_stack = torch.stack([p[3] for p in pages], dim=0)  # (P, H, D)
        dmin = (q2.unsqueeze(0) * mn_stack).sum(-1)  # (P, H)
        dmax = (q2.unsqueeze(0) * mx_stack).sum(-1)  # (P, H)
        ub = torch.maximum(dmin, dmax).sum(-1)       # (P,)

        if self.sram_k.shape[1] > 0:
            s_vram = torch.matmul(q, self.sram_k.transpose(-2, -1)).squeeze(1)
            # Threshold against the MEDIAN current VRAM relevance, not the
            # min. The min is almost always the most recently arrived
            # occupant (grace-protected, cum=0, unproven) -- comparing new
            # candidates against that single weakest, still-unproven
            # resident creates a self-reinforcing promote/evict feedback
            # loop (each promotion creates a new "weakest occupant" that
            # the next candidate trivially beats), causing runaway churn
            # that was measured to promote/demote nearly every token in
            # VRAM every single step with ~0% of promotions surviving long
            # enough to be useful. The median is a page's bar against a
            # "typical" resident, not the most fragile one.
            thresh = s_vram.sum(0).median()
        else:
            thresh = torch.tensor(float("-inf"), device=ub.device)

        k = min(self.cfg.promote_top_pages, ub.shape[0])
        top_vals, top_idx = torch.topk(ub, k)
        keep = (top_vals > thresh).tolist()
        top_idx = top_idx.tolist()
        hits = [pages[i][:2] for i, k_ in zip(top_idx, keep) if k_]
        sketch_macs = 2 * H * len(pages) * D
        return hits, sketch_macs

    # =====================================================================
    # STEP 1b : peek vs. promote -- hysteresis on RE-entry only
    # =====================================================================
    _HYSTERESIS_MIN_STREAK = 2  # smallest value that means "more than
    # once" -- the structural minimum for "requires repeated candidacy",
    # not a tuned hyperparameter. See resolve_promotions() docstring.

    def resolve_promotions(self, candidates):
        """Split sketch-check candidates into (promote_now, peek_only).

        Promotion currently pays the full cost of a residency move (VRAM
        slot contention, later write-back) just to let the model glance at
        a token once -- measured to be the dominant source of promote/evict
        churn (near-max promotion rate, most of it single-use). This
        separates VISIBILITY from RESIDENCY:

        - A candidate page that has NEVER been VRAM-resident before
          promotes immediately on its first candidacy -- identical to the
          pre-peek/promote behavior, so a token gets its first chance at
          VRAM exactly as before.
        - A candidate page that WAS evicted from VRAM before (it already
          had its chance and proved, at least once, not durably valuable
          enough to keep) must clear the sketch bar for
          _HYSTERESIS_MIN_STREAK consecutive steps before re-promoting.
          This directly targets the measured churn pattern: promote, one
          glance, evict, immediately re-qualify, repeat.
        - Anything that doesn't clear the promotion bar is still returned
          as a peek candidate: get_peek_kv() + compute_attention() let the
          model attend to it in place this step (real exact attention, not
          just the sketch's upper bound), with no residency change, no
          write, no eviction pressure. Nearly free -- it's HAM already
          being paid for by the sketch check.

        Deliberately NOT a self-tuning threshold (e.g. an ARC-style
        adaptive split): a purely reactive "calibrate the bar from past
        promotions" scheme deadlocks here, since gating pages to a streak
        of 1 forces every recorded promotion to have streak exactly 1,
        which then re-derives a threshold of 1 forever. `2` is left as an
        explicit structural minimum (the smallest value that means
        "repeated" at all) rather than papering over that with a fabricated
        adaptive number; a real adaptive scheme would need a calibration
        phase decoupled from the gating decision, which is future work.
        """
        m = self.stt_k.shape[1]
        if m == 0 or not candidates:
            return [], []

        candidate_idx = set()
        for (s, e) in candidates:
            candidate_idx.update(i for i in range(s, e) if not self.stt_shadow[i])

        in_cand = torch.zeros(m, dtype=torch.bool, device=self.stt_peek_streak.device)
        if candidate_idx:
            in_cand[list(candidate_idx)] = True
        self.stt_peek_streak = torch.where(
            in_cand, self.stt_peek_streak + 1, torch.zeros_like(self.stt_peek_streak)
        )

        promote_hits, peek_hits = [], []
        for (s, e) in candidates:
            real_idx = [i for i in range(s, e) if not self.stt_shadow[i]]
            if not real_idx:
                continue
            needs_gate = any(self.stt_was_vram_resident[i] for i in real_idx)
            streak_here = self.stt_peek_streak[real_idx].min().item()
            if needs_gate and streak_here < self._HYSTERESIS_MIN_STREAK:
                peek_hits.append((s, e))
            else:
                promote_hits.append((s, e))
        return promote_hits, peek_hits

    def get_peek_kv(self, peek_hits):
        """Gather (but do not move) K/V for peek-only candidate pages, for
        use in this step's attention alongside the VRAM working set. No
        residency change, no write, no eviction pressure -- see
        resolve_promotions() docstring.

        Returns (k_peek, v_peek, peek_pos) where peek_pos is each peeked
        token's TRUE ORIGINAL DOCUMENT POSITION (stable identity), not its
        raw array index -- promote() can reindex/shrink the STT arrays
        (non-inclusive mode) between this call and any later use of these
        tokens' indices, so callers must re-resolve position -> current
        index afterward rather than reusing a raw index across that call.
        """
        idx = []
        for (s, e) in peek_hits:
            idx += [i for i in range(s, e) if not self.stt_shadow[i]]
        if not idx:
            return None, None, []
        peek_pos = [self.stt_pos[i] for i in idx]
        return self.stt_k[:, idx, :], self.stt_v[:, idx, :], peek_pos

    # =====================================================================
    # STEP 2 : promote (STT-RAM -> VRAM)
    # =====================================================================
    def promote(self, hits):
        """Move real-victim tokens in hit pages STT-RAM -> VRAM."""
        if not hits:
            return 0

        H, m, D = self.stt_k.shape
        idx = []
        for (s, e) in hits:
            idx += [i for i in range(s, e) if not self.stt_shadow[i]]
        if not idx:
            return 0

        self.sram_k = torch.cat([self.sram_k, self.stt_k[:, idx, :]], dim=1)
        self.sram_v = torch.cat([self.sram_v, self.stt_v[:, idx, :]], dim=1)
        self.sram_pos += [self.stt_pos[i] for i in idx]
        self.sram_cum = torch.cat(
            [self.sram_cum, torch.zeros(len(idx), device=self.cfg.device, dtype=self.cfg.dtype)]
        )
        self.sram_age = torch.cat(
            [self.sram_age, torch.zeros(len(idx), device=self.cfg.device, dtype=self.cfg.dtype)]
        )
        # Promoted tokens enter with cum=0, which would make them the very
        # next eviction candidate before compute_attention() ever gives them
        # a chance to be attended to. Grace-protect them for exactly one
        # evict_and_demote() pass (cleared at the end of that pass) so they
        # compete on REAL accumulated attention afterward, not a fabricated
        # score. Structurally identical to sink/window protection.
        self.sram_grace += [True] * len(idx)

        # This token is now (or again) VRAM-resident -- mark it so, so a
        # future eviction back to STT-RAM knows to gate its re-promotion
        # (see resolve_promotions). Idempotent for tokens promoted before.
        for i in idx:
            self.stt_was_vram_resident[i] = True

        if self.cfg.inclusive:
            for i in idx:
                self.stt_shadow[i] = True
                self.stt_last_access[i] = self._t
        else:
            drop = set(idx)
            keep_idx = [i for i in range(m) if i not in drop]
            self.stt_k = self.stt_k[:, keep_idx, :]
            self.stt_v = self.stt_v[:, keep_idx, :]
            self.stt_pos = [self.stt_pos[i] for i in keep_idx]
            self.stt_last_access = self.stt_last_access[keep_idx]
            self.stt_shadow = [self.stt_shadow[i] for i in keep_idx]
            self.stt_was_vram_resident = [self.stt_was_vram_resident[i] for i in keep_idx]
            self.stt_peek_streak = self.stt_peek_streak[keep_idx]

        return len(idx)

    # =====================================================================
    # STEP 3 : compute attention over VRAM working set
    # =====================================================================
    def compute_attention(self, q, peek_k=None, peek_v=None):
        """Attention over the VRAM working set, optionally extended with
        peeked STT-RAM pages (see resolve_promotions/get_peek_kv) for this
        step only -- peeked tokens contribute to the real output and get
        real exact-attention feedback, without becoming VRAM-resident.
        With peek_k=None this is bit-identical to the pre-peek/promote
        behavior.
        """
        n_sram = self.sram_k.shape[1]
        if peek_k is not None and peek_k.shape[1] > 0:
            full_k = torch.cat([self.sram_k, peek_k], dim=1)
            full_v = torch.cat([self.sram_v, peek_v], dim=1)
        else:
            full_k, full_v = self.sram_k, self.sram_v

        out, attn_w, macs = scaled_dot_product_attention(q, full_k, full_v)
        attn_flat = attn_w.squeeze(1)  # (H, Lk_total)
        self.sram_cum += attn_flat[:, :n_sram].sum(0)
        self.sram_age += 1
        peek_attn = attn_flat[:, n_sram:].sum(0) if full_k.shape[1] > n_sram else None
        return out, macs, peek_attn

    # =====================================================================
    # STEP 4 : evict + demote
    # =====================================================================
    def _hard_protected_mask(self):
        """True where a VRAM token must NEVER be evicted (sink / recent window)."""
        n = len(self.sram_pos)
        if n == 0:
            return torch.zeros(0, dtype=torch.bool)
        cfg = self.cfg
        pos_t = torch.tensor(self.sram_pos)
        max_pos = pos_t.max()
        return (pos_t < cfg.sink_size) | (pos_t > max_pos - cfg.window_size)

    def evict_and_demote(self, new_k, new_v, new_pos):
        cfg = self.cfg
        dropped = 0
        demoted = 0
        writes_saved = 0
        reclaimed = 0

        # Add new token to VRAM
        self.sram_k = torch.cat([self.sram_k, new_k], dim=1)
        self.sram_v = torch.cat([self.sram_v, new_v], dim=1)
        self.sram_pos.append(new_pos)
        self.sram_cum = torch.cat(
            [self.sram_cum, torch.zeros(1, device=cfg.device, dtype=cfg.dtype)]
        )
        self.sram_age = torch.cat(
            [self.sram_age, torch.zeros(1, device=cfg.device, dtype=cfg.dtype)]
        )
        self.sram_grace.append(False)  # newest token is protected via window, not grace

        # VRAM overflow -> demote the unprotected token with the lowest
        # AVERAGE attention per attended step (sram_cum / sram_age), not
        # raw cumulative attention. Raw cumulative sum structurally favors
        # whichever token has been resident longest: a token promoted 1
        # step ago cannot out-accumulate one that has been sitting in VRAM
        # for 500 steps even if the newcomer is now clearly more relevant,
        # which was measured to make every promoted token the eviction
        # candidate again the moment its one-step grace period ends --
        # permanent thrashing that starves the write-saving mechanism.
        # Averaging by age compares tokens on relevance-per-step instead of
        # tenure. Ties (age=0, i.e. not yet attended even once) fall back
        # to raw cum via the +1 floor, matching the pre-existing behavior
        # for brand-new/just-promoted tokens.
        #
        # Preference order: ordinary tokens first; if none remain (e.g. a
        # promotion just added more grace-protected tokens than there is
        # room for), fall back to evicting a grace token -- capacity is a
        # hard constraint, grace is only a preference. Sink/window tokens
        # are never evicted.
        # Victim SELECTION is batched: none of the candidates' scores change
        # as others are (conceptually) removed within this call -- sram_cum
        # and sram_age are read-only here, so picking the N lowest-scoring
        # unprotected tokens in one shot gives the IDENTICAL set (and, via a
        # stable sort, identical tie-breaking order) as the old one-at-a-time
        # argmin loop, without rebuilding an O(n) Python mask and reallocating
        # the (H, n, D) VRAM tensor on every single evicted token. Measured
        # to be >90% of this function's wall-clock time before batching
        # (32 layers x up to ~32 evictions/layer/step, each doing a full
        # tensor-copy round trip).
        n_over = len(self.sram_pos) - cfg.sram_capacity
        victims = []
        if n_over > 0:
            hard = self._hard_protected_mask()
            grace_t = torch.tensor(self.sram_grace, dtype=torch.bool)
            soft = hard | grace_t
            cand = self.sram_cum / self.sram_age.clamp(min=1)

            cand_soft = cand.clone()
            cand_soft[soft] = float("inf")
            order = torch.argsort(cand_soft, stable=True)
            n_avail = int(torch.isfinite(cand_soft).sum().item())
            take = min(n_over, n_avail)
            victims = order[:take].tolist()

            remaining = n_over - take
            if remaining > 0:
                # Not enough ordinary candidates (e.g. a promotion just added
                # more grace tokens than there is room for): capacity is a
                # hard constraint, grace is only a preference, so fall back
                # to evicting grace tokens too. Sink/window are never touched.
                cand_hard = cand.clone()
                cand_hard[hard] = float("inf")
                already = torch.zeros(len(self.sram_pos), dtype=torch.bool)
                already[victims] = True
                cand_hard[already] = float("inf")
                order_hard = torch.argsort(cand_hard, stable=True)
                n_hard_avail = int(torch.isfinite(cand_hard).sum().item())
                take_hard = min(remaining, n_hard_avail)
                victims += order_hard[:take_hard].tolist()
                # if still short, everything left is sink/window -- matches
                # the old loop's "break": cannot evict further.

        # Backup lookup: build the pos->index map ONCE (O(m)) instead of
        # calling _find_backup's linear scan once per victim (O(m) each,
        # O(m * victims) total). Safe to compute up front and treat as
        # read-only for this batch: sram_pos entries are unique, so no two
        # victims can share a vpos / contend for the same backup slot.
        backup_by_pos = {}
        if cfg.inclusive and self.stt_shadow:
            shadow_idx0 = torch.where(torch.tensor(self.stt_shadow, dtype=torch.bool))[0].tolist()
            backup_by_pos = {self.stt_pos[i]: i for i in shadow_idx0}

        no_backup_victims = []
        for victim in victims:
            demoted += 1
            vpos = self.sram_pos[victim]
            back = backup_by_pos.get(vpos, -1)
            if back >= 0:
                self.stt_shadow[back] = False
                self.stt_last_access[back] = self._t
                # Fresh re-candidacy episode -- was_vram_resident is already
                # True (that's why it has a backup), streak must not carry
                # over a stale value frozen from before it became a shadow.
                self.stt_peek_streak[back] = 0
                writes_saved += 1
            else:
                no_backup_victims.append(victim)

        if no_backup_victims:
            # One batched STT append for every victim with no existing
            # backup, instead of one torch.cat (full tensor reallocation)
            # per victim.
            self.stt_k = torch.cat([self.stt_k, self.sram_k[:, no_backup_victims, :]], dim=1)
            self.stt_v = torch.cat([self.stt_v, self.sram_v[:, no_backup_victims, :]], dim=1)
            self.stt_pos += [self.sram_pos[v] for v in no_backup_victims]
            self.stt_last_access = torch.cat([
                self.stt_last_access,
                torch.full((len(no_backup_victims),), float(self._t),
                           device=cfg.device, dtype=cfg.dtype),
            ])
            self.stt_shadow += [False] * len(no_backup_victims)
            # These tokens just left VRAM -- any future re-promotion is a
            # RE-entry and must clear the hysteresis gate, not the free pass
            # a genuinely-new candidate gets.
            self.stt_was_vram_resident += [True] * len(no_backup_victims)
            self.stt_peek_streak = torch.cat([
                self.stt_peek_streak,
                torch.zeros(len(no_backup_victims), device=cfg.device, dtype=cfg.dtype),
            ])

        if victims:
            keep_mask = torch.ones(len(self.sram_pos), dtype=torch.bool)
            keep_mask[victims] = False
            keep = torch.where(keep_mask)[0].tolist()
            self.sram_k = self.sram_k[:, keep, :]
            self.sram_v = self.sram_v[:, keep, :]
            self.sram_pos = [self.sram_pos[i] for i in keep]
            self.sram_cum = self.sram_cum[keep_mask]
            self.sram_age = self.sram_age[keep_mask]
            self.sram_grace = [self.sram_grace[i] for i in keep]

        # STT-RAM overflow -> reclaim shadows first, then LRU evict.
        #
        # Reclaim priority among shadows is NOT last-access LRU: a shadow's
        # last_access is stamped once at promotion time and never updated
        # while its twin sits happily in VRAM, so LRU-by-age reclaims
        # long-resident (i.e. currently IMPORTANT, high cum-attn) tokens'
        # backups first -- exactly backwards. Instead reclaim the shadow
        # whose live VRAM twin currently has the HIGHEST cumulative
        # attention: that token is the least likely to be demoted again
        # soon, so losing its free backup costs the least expected future
        # writes. A shadow whose twin can't be found (shouldn't happen --
        # invariant) is reclaimed first.
        # Batched for the same reason as the VRAM loop above: a candidate's
        # reclaim/eviction priority here (its live twin's cum-attn for
        # shadows, or last_access for real victims) does not change as
        # other candidates are removed, so the N needed removals can be
        # selected in one shot instead of one Python while-loop iteration
        # (with a full dict rebuild) per removed token.
        n_over = len(self.stt_pos) - cfg.sttram_capacity
        if n_over > 0:
            # One boolean tensor pass instead of two separate Python
            # enumerate() loops over stt_shadow -- with STT-RAM near
            # capacity almost every step, this ran O(sttram_capacity)
            # Python-level work per layer per step and was measured to
            # dominate real wall-clock time once STT-RAM got large.
            shadow_t = torch.tensor(self.stt_shadow, dtype=torch.bool)
            shadow_idx = torch.where(shadow_t)[0].tolist()
            evict_set = []

            if shadow_idx:
                sram_cum_by_pos = dict(zip(self.sram_pos, self.sram_cum.tolist()))
                priority = torch.tensor(
                    [sram_cum_by_pos.get(self.stt_pos[i], float("inf")) for i in shadow_idx]
                )
                order = torch.argsort(priority, descending=True, stable=True)
                take = min(n_over, len(shadow_idx))
                evict_set = [shadow_idx[i] for i in order[:take].tolist()]
                reclaimed += take

            remaining = n_over - len(evict_set)
            if remaining > 0:
                real_idx = torch.where(~shadow_t)[0].tolist()
                la = self.stt_last_access[real_idx]
                order = torch.argsort(la, stable=True)
                take = min(remaining, len(real_idx))
                real_evict = [real_idx[i] for i in order[:take].tolist()]
                if cfg.store_dram:
                    for i in real_evict:
                        self.dram_k = torch.cat([self.dram_k, self.stt_k[:, i:i + 1, :]], dim=1)
                        self.dram_v = torch.cat([self.dram_v, self.stt_v[:, i:i + 1, :]], dim=1)
                        self.dram_pos.append(self.stt_pos[i])
                dropped += len(real_evict)
                evict_set += real_evict

            keep_mask = torch.ones(len(self.stt_pos), dtype=torch.bool)
            keep_mask[evict_set] = False
            keep = torch.where(keep_mask)[0].tolist()
            self.stt_k = self.stt_k[:, keep, :]
            self.stt_v = self.stt_v[:, keep, :]
            self.stt_pos = [self.stt_pos[i] for i in keep]
            self.stt_last_access = self.stt_last_access[keep_mask]
            self.stt_shadow = [self.stt_shadow[i] for i in keep]
            self.stt_was_vram_resident = [self.stt_was_vram_resident[i] for i in keep]
            self.stt_peek_streak = self.stt_peek_streak[keep_mask]

        # Grace protection is one-shot: it only shields a just-promoted token
        # through the eviction pass immediately following its own promotion.
        self.sram_grace = [False] * len(self.sram_grace)

        return demoted, dropped, writes_saved, reclaimed

    # =====================================================================
    # Full decode step orchestrator
    # =====================================================================
    def step(self, q, new_k, new_v, new_pos):
        self._t += 1
        H, D = self.cfg.num_heads, self.cfg.head_dim
        rec = StepRecord(step=self._t)

        # 1. Sketch check -> peek/promotion candidates
        candidates, sketch_macs = self.sketch_check(q)
        rec.sketch_macs = sketch_macs

        # 1b. Hysteresis: first-time candidates promote immediately (as
        # before); re-candidates (previously evicted from VRAM) must clear
        # the bar repeatedly first -- see resolve_promotions() docstring.
        promote_hits, peek_hits = self.resolve_promotions(candidates)

        # 2a. Peek (read-only, no residency change) -- gather still-candidate
        # pages for this step's attention only. Must happen BEFORE promote()
        # below: in non-inclusive mode, promote() removes indices from the
        # STT arrays and everything after it shifts, which would leave
        # peek_hits' (start, end) slices pointing at the wrong -- or
        # out-of-range -- tokens. promote_hits/peek_hits are always disjoint
        # page ranges, so gathering peek data first is always safe.
        peek_k, peek_v, peek_pos = self.get_peek_kv(peek_hits)
        rec.peeked_tokens = len(peek_pos)

        # 2b. Promote (physical move -- VRAM slot + eventual write-back cost)
        rec.promoted_tokens = self.promote(promote_hits)

        # 3. Attention on VRAM working set (+ peeked STT pages, if any)
        out, attn_macs, peek_attn = self.compute_attention(q, peek_k, peek_v)
        rec.attn_macs = attn_macs

        if peek_pos and peek_attn is not None:
            # Real exact-attention feedback (not just the sketch bound):
            # a peeked-but-not-promoted token that actually got attended to
            # is marked freshly relevant, protecting it from LRU-based STT
            # reclaim a bit longer -- reuses the existing stt_last_access
            # mechanism rather than inventing a new signal. Re-resolve
            # position -> current array index since promote() above may
            # have reindexed the STT arrays (non-inclusive mode).
            touched_pos = [p for p, a in zip(peek_pos, peek_attn.tolist()) if a > 0]
            if touched_pos:
                pos_to_idx = {p: i for i, p in enumerate(self.stt_pos)}
                touched_idx = [pos_to_idx[p] for p in touched_pos if p in pos_to_idx]
                if touched_idx:
                    self.stt_last_access[touched_idx] = self._t

        # 4. Evict + demote (also inserts new token)
        demoted, dropped, writes_saved, reclaimed = self.evict_and_demote(new_k, new_v, new_pos)

        rec.demoted_tokens = demoted
        rec.dropped_tokens = dropped
        rec.writes_saved = writes_saved
        rec.backups_reclaimed = reclaimed

        rec.sram_tokens = len(self.sram_pos)
        rec.sttram_tokens = len(self.stt_pos)
        rec.dram_tokens = len(self.dram_pos)

        # Cost model
        paid_writes = rec.demoted_tokens - rec.writes_saved
        elems_per_tok = H * D
        lat_p, eng_p = self.cm.promote_cost(rec.promoted_tokens * 2 * elems_per_tok)
        lat_d, eng_d = self.cm.demote_cost(paid_writes * 2 * elems_per_tok)
        lat_dd, eng_dd = self.cm.deep_demote_cost(rec.dropped_tokens * 2 * elems_per_tok)
        lat_vr, eng_vr = self.cm.gpu_vram_read_cost(rec.sram_tokens * 2 * elems_per_tok)
        # Peek: a real STT-RAM read for exact attention, no VRAM write --
        # the whole point is that this is cheaper than promote_cost.
        lat_pk, eng_pk = self.cm.read_latency_us("STT-RAM", rec.peeked_tokens * 2 * elems_per_tok), \
                         self.cm.read_energy_nj("STT-RAM", rec.peeked_tokens * 2 * elems_per_tok)

        n_pages = math.ceil(max(rec.sttram_tokens, 1) / self.cfg.page_size)
        lat_sk, eng_sk = self.cm.sketch_read_cost(2 * n_pages * H * D, tier="STT-RAM")

        rec.lat_sketch_us = lat_sk
        rec.lat_promote_us = lat_p
        rec.lat_peek_us = lat_pk
        rec.lat_attention_us = lat_vr
        rec.lat_demote_us = lat_d + lat_dd
        rec.latency_us = lat_sk + lat_p + lat_pk + lat_vr + lat_d + lat_dd
        rec.energy_nj = eng_sk + eng_p + eng_pk + eng_vr + eng_d + eng_dd

        self.metrics.add(rec)
        return out